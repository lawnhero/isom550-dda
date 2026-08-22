import time
import uuid
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.globals import set_verbose

import utils.chains_lcel as chains
import utils.ui as ui
from utils.agent_graph import build_ta_agent, run_ta_turn
from utils.attachments import (
    IMAGE_LIMIT_NOTICE,
    MAX_IMAGES,
    describe_images,
    extract_attachments,
)
from utils.course_context import (
    get_course_context,
    get_course_date_span,
    get_course_links,
    get_software_context,
)
from utils.progress import PHASE_FAILED, PHASE_WRITING, ProgressReporter
from utils.sidebar import sidebar, update_session_stats
import utils.llm_models as llms
from utils.ta_tools import TurnArtifacts

# Set the page_title
st.set_page_config(
    page_title="ISOM 550 DDA Virtual TA",
    page_icon=":material/school:",
    layout="wide",
)

# cache the vectorized embedding database
from utils.utils import (
    load_db,
    query_db_connection,
    build_event_payload,
    store_event,
    store_feedback,
)

# Disable verbose chain debug logs in normal app use.
set_verbose(False)

# 1. Load the Vectorised database
#
# Tier A is deliberately absent here: course facts (dates, people, grading) are
# looked up from course_data/ by utils.course_context, not embedded. There used
# to be a `data/course` index of the same facts as Q&A rows; nothing had queried
# it since Tier A landed, and it had gone stale enough to contradict facts.toml
# (it still claimed Spring 2026 office hours), so it is gone rather than dormant.
contents_path = 'data/concepts'
documents_path = 'data/documents'
# Tier B: concept index from course_data/concepts.toml (scripts/build_concepts.py).
contents_db = load_db(db_path=contents_path, label='concepts')
# Tier C: class recaps + assignment briefs, built by scripts/build_documents.py.
# Missing until that runs; searches then return nothing and the tool abstains.
documents_db = load_db(db_path=documents_path, label='assignments and recaps')

# 2. MongoDB Atlas connection
mongo_db = query_db_connection()
collection = mongo_db['ISOM 550']

# 3. Setup LLM and chains
main_tutor = llms.deepseekv4_with_grok_fallback
claude_haiku = llms.deepseek_v4_flash
agent_llm = llms.openai_gpt56_luna

all_chains = chains.get_all_chains(
    main_tutor,
    claude_haiku,
    # Screenshot turns are pinned here regardless of which model is
    # tutoring: reading a regression table out of a PNG is a different
    # capability from writing the explanation, and it should not change
    # every time the primary tutor is swapped.
    vision_llm=llms.openai_gpt56_luna_vision,
)
class_chain = all_chains['class_chain']

TA_AVATAR = ":material/school:"

# How many student turns between recap cards.
RECAP_EVERY = 8


def _parse_chat_input(raw_input):
    """Normalize st.chat_input return value (str or ChatInputValue)."""
    if raw_input is None:
        return "", []
    if isinstance(raw_input, str):
        return raw_input.strip(), []
    text = (getattr(raw_input, "text", None) or raw_input.get("text") or "").strip()
    files = list(getattr(raw_input, "files", None) or raw_input.get("files") or [])
    return text, files


def _status_sink(status):
    """Paint one progress event onto the live st.status container.

    The label is the headline -- what the tutor is doing right now -- and the
    detail lines accumulate inside the (collapsed) container as a record of how
    the answer was actually put together: which tool the router picked, what it
    searched for, how many passages came back, which sources they were.

    Phrasing for tool phases comes from ui.working_label, i.e. the same table as
    the provenance badge under the finished answer, so the two cannot drift.

    Debug events (tool names, raw router arguments) are dropped unless
    diagnostics are on: a student reading "Chose answer_concept (query=...)"
    learns nothing the next line does not say in plain language.
    """
    def sink(event):
        if event.debug and not st.session_state.get("show_diagnostics"):
            return
        label = event.label or ui.working_label(event.tools)
        if label:
            suffix = "..." if event.state == "running" else ""
            status.update(label=f"{label}{suffix}", state=event.state)
        if event.detail:
            status.caption(event.detail)

    return sink


def _stream_answer(stream, progress=None, label=PHASE_WRITING):
    """Stream tutor text; announce the phase on the first token."""
    if progress is None:
        return ui.write_stream_md(stream)

    placeholder = st.empty()
    chunks = []
    for chunk in stream:
        if not chunks:
            progress.emit(label="Preparing your answer...")
        else:
            progress.emit(label=label)
        text = chunk if isinstance(chunk, str) else getattr(chunk, "content", None) or str(chunk)
        chunks.append(text)
        placeholder.markdown(ui.escape_md_dollars("".join(chunks)))
    return "".join(chunks)


# --------------------------------------------------------------------------
# Per-message metadata
#
# Keyed by index into chat_history. Everything a finished turn needs to be
# redrawn after the closing st.rerun() lives here: provenance, sources,
# abstention, feedback id, recap. Before this existed those were rendered live
# and then destroyed by the rerun -- sources and recap cards in particular were
# computed on every turn and seen by nobody.
# --------------------------------------------------------------------------
def _set_meta(index: int, **fields) -> None:
    st.session_state.message_meta.setdefault(index, {}).update(fields)


def _get_meta(index: int) -> dict:
    return st.session_state.message_meta.get(index, {})


def _record_feedback(interaction_id: str, key: str) -> None:
    """st.feedback callback: log the rating without a second rerun."""
    value = st.session_state.get(key)
    if value is None:
        return
    store_feedback(
        collection=collection,
        session_id=st.session_state.session_id,
        interaction_id=interaction_id,
        helpful="Helpful" if value == 1 else "Not helpful",
        note="",
        mode="unified",
    )
    st.session_state.feedback_submitted_ids.append(interaction_id)
    st.toast("Thanks — that helps improve the tutor.", icon=":material/favorite:")


def _render_section(section: dict, *, key: str) -> None:
    """One answer section: its text, its badge, its sources."""
    ui.md(section.get("text", ""))
    weak = section.get("retrieval_quality") == "weak"
    ui.render_provenance(
        section.get("route", ""),
        abstained=section.get("abstained", False),
        weak=weak,
    )
    ui.render_sources(section.get("sources") or [], key=f"sources_{key}", expanded=weak)
    if section.get("abstained"):
        ui.render_unresolved(get_course_links(), key=f"unresolved_{key}")


def _render_ai_message(index: int, content: str, *, is_last: bool) -> None:
    """Draw one assistant turn plus everything hanging off it."""
    meta = _get_meta(index)
    with st.chat_message("AI", avatar=TA_AVATAR):
        # Sits where the live status sat, in the state it finished in.
        ui.render_progress(meta.get("progress"))

        if meta.get("attachment_notice"):
            st.warning(meta["attachment_notice"], icon=":material/image_not_supported:")

        sections = meta.get("sections") or []
        if sections:
            # Redraw the turn the way it streamed: each tool's answer with its
            # own badge, rather than one badge over everything.
            for position, section in enumerate(sections):
                if position:
                    st.space("small")
                _render_section(section, key=f"{index}_{position}")
        else:
            # Greeting, clarifying turn, or the ungrounded fallback answer.
            ui.md(content)
            weak = meta.get("retrieval_quality") == "weak"
            if meta.get("route"):
                ui.render_provenance(
                    meta["route"],
                    abstained=meta.get("abstained", False),
                    weak=weak,
                )
            ui.render_sources(
                meta.get("sources") or [], key=f"sources_{index}", expanded=weak
            )
            if meta.get("abstained"):
                ui.render_unresolved(get_course_links(), key=f"unresolved_{index}")

        ui.render_recap(meta.get("recap", ""))

        interaction_id = meta.get("interaction_id")
        if is_last and interaction_id:
            if interaction_id in st.session_state.feedback_submitted_ids:
                st.caption("Rating recorded.")
            else:
                key = f"feedback_{interaction_id}"
                st.feedback(
                    "thumbs",
                    key=key,
                    on_change=_record_feedback,
                    args=(interaction_id, key),
                )

        if is_last and st.session_state.get("show_diagnostics") and meta.get("diagnostics"):
            ui.render_diagnostics(meta["diagnostics"], key=f"diag_{index}")


# --------------------------------------------------------------------------
# Clarifying turns
# --------------------------------------------------------------------------
def _append_clarifying_turn(action_label: str, clarify_text: str, intent: str, needs: str):
    """Record a clarifying turn and wait for the missing topic/attempt."""
    st.session_state.chat_history.append(HumanMessage(action_label))
    st.session_state.chat_history.append(AIMessage(clarify_text))
    st.session_state.pending_intent = {
        "intent": intent,
        "label": action_label,
        "needs": needs,
    }


def _cancel_pending() -> None:
    st.session_state.pending_intent = None
    st.session_state.pop("clarify_topic_pills", None)
    st.session_state.pop("clarify_subtopic_pills", None)


def _conversation_started(chat_history) -> bool:
    """True once the student has sent at least one message."""
    return any("Human" in str(type(message)) for message in (chat_history or []))


def _resolve_action(action: dict):
    """Resolve a clicked chip into (user_query, display_text, did_clarify).

    did_clarify=True means we asked a question instead and should rerun.
    """
    if action.get("kind") == "query":
        return action["value"], action["label"], False

    intent = action["value"]

    if intent in {"explain", "practice", "check"}:
        _append_clarifying_turn(
            action["label"], action["clarify"], intent, action["needs"]
        )
        return "", "", True

    topic = (st.session_state.get("last_practice_topic") or "").strip()
    if not topic:
        topic = chains.infer_topic_from_history(st.session_state.chat_history)

    if intent == "practice_same":
        if topic:
            return chains.compose_quick_action_query("practice", topic=topic), action["label"], False
        _append_clarifying_turn(
            action["label"],
            "What would you like to practice? Pick a topic below or type it in the chat.",
            "practice",
            "topic",
        )
        return "", "", True

    if intent == "practice_harder":
        if topic:
            return (
                f"Generate a harder practice question on this ISOM 550 topic: {topic}. "
                "Stay strictly on this topic; do not invent an unrelated scenario."
            ), action["label"], False
        _append_clarifying_turn(
            action["label"],
            "What topic should the harder practice question focus on?",
            "practice",
            "topic",
        )
        return "", "", True

    return "", "", False


def _advance_to_subtopic_selection(pending: dict, parent_topic: str):
    """Move from topic selection to subtopic pills for the chosen parent topic."""
    parent_topic = (parent_topic or "").strip()
    clarify = (
        f"Which part of **{parent_topic}** do you want to focus on? "
        "Pick a subtopic below or type it in the chat."
    )
    st.session_state.chat_history.append(HumanMessage(parent_topic))
    st.session_state.chat_history.append(AIMessage(clarify))
    st.session_state.pending_intent = {
        "intent": pending["intent"],
        "label": pending.get("label", parent_topic),
        "needs": "subtopic",
        "parent_topic": parent_topic,
    }
    st.session_state.pop("clarify_topic_pills", None)
    st.session_state.pop("clarify_subtopic_pills", None)


def _resolve_pending_intent(pending, selected_value, typed_query, uploaded_files):
    """
    Resolve a pending clarify step.

    Returns (user_query, escaped, did_clarify_again).
      escaped=True          -> the student typed a new question instead of an
                               answer; abandon the pending intent and treat the
                               text as an ordinary question.
      did_clarify_again=True -> we advanced topic -> subtopic and should rerun.
    """
    intent = pending["intent"]
    needs = pending["needs"]

    if needs in {"topic", "subtopic"}:
        choice = (selected_value or typed_query or "").strip()
        if not choice:
            return "", False, False

        # A pill click is always an answer to the clarifying question. Only
        # typed text can be a change of subject.
        if not selected_value and chains.is_new_question(typed_query):
            return typed_query, True, False

        if needs == "topic":
            # Pill selection (or typed exact topic label) with subtopics -> ask subtopic next.
            if choice in chains.CURRICULUM_TOPICS and chains.get_subtopics(choice):
                _advance_to_subtopic_selection(pending, choice)
                return "", False, True
            return chains.compose_quick_action_query(intent, topic=choice), False, False

        focus = chains.format_topic_focus(pending.get("parent_topic", ""), choice)
        return chains.compose_quick_action_query(intent, topic=focus), False, False

    if needs == "attempt":
        if not typed_query and not uploaded_files:
            return "", False, False
        # A question with nothing attached is not an attempt to grade. Without
        # this, "when is A3 due?" became "Please check my attempt: when is A3 due?"
        if not uploaded_files and chains.is_new_question(typed_query):
            return typed_query, True, False
        return chains.compose_quick_action_query(intent, attempt_text=typed_query), False, False

    return typed_query, False, False


# 4. Build an app with streamlit
def main():
    st.session_state.setdefault("session_id", str(uuid.uuid4()))
    st.session_state.setdefault("feedback_submitted_ids", [])
    st.session_state.setdefault("pending_intent", None)
    st.session_state.setdefault("last_practice_topic", "")
    st.session_state.setdefault("message_meta", {})
    st.session_state.setdefault("student_turns", 0)

    st.title("ISOM 550 Virtual TA")
    sidebar_settings = sidebar()
    initial_text = (
        "Hi, I'm Dayton, your virtual TA."
    )

    # Initialize chat history in session state
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = [AIMessage(initial_text)]

    # display previous conversation history
    last_index = len(st.session_state.chat_history) - 1
    for idx, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                ui.md(message.content)
        elif isinstance(message, AIMessage):
            _render_ai_message(idx, message.content, is_last=(idx == last_index))

    pending = st.session_state.pending_intent
    conversation_started = _conversation_started(st.session_state.chat_history)

    # Starter prompts fill the empty main area before the first question.
    starter_choice = None
    if not conversation_started and not pending:
        st.caption("Try asking")
        starter_choice = st.pills(
            "Example questions",
            options=ui.STARTER_PROMPTS,
            selection_mode="single",
            key="starter_prompt_pills",
            label_visibility="collapsed",
        )

    # Topic / subtopic pills while waiting for a clarifying detail.
    selected_topic = None
    selected_subtopic = None
    if pending and pending.get("needs") == "topic":
        st.caption("Suggested topics")
        selected_topic = st.pills(
            "Course topics",
            options=chains.CURRICULUM_TOPICS,
            selection_mode="single",
            key="clarify_topic_pills",
            label_visibility="collapsed",
        )
    elif pending and pending.get("needs") == "subtopic":
        parent = pending.get("parent_topic", "")
        subtopics = chains.get_subtopics(parent)
        st.caption(f"Subtopics in {parent}" if parent else "Suggested subtopics")
        if subtopics:
            selected_subtopic = st.pills(
                "Course subtopics",
                options=subtopics,
                selection_mode="single",
                key="clarify_subtopic_pills",
                label_visibility="collapsed",
            )

    # An explicit way out of the clarify state. Typing a new question also
    # works (see _resolve_pending_intent), but the student should be able to
    # see that backing out is allowed rather than having to discover it.
    if pending:
        with st.container(horizontal=True, horizontal_alignment="left"):
            if st.button(
                "Never mind",
                icon=":material/close:",
                key="cancel_pending",
                type="tertiary",
            ):
                _cancel_pending()
                st.rerun()

    # Pin composer controls so students always see them while scrolling history.
    selected_action = None
    with st.bottom:
        if not pending:
            if not conversation_started:
                st.caption("Quick actions")
                selected_action = ui.render_action_row(
                    ui.QUICK_ACTIONS, key_prefix="quick_action"
                )
            else:
                # Chips follow the route that answered the last turn: a
                # deadline answer offers "what's due next", a practice question
                # offers "check my work". A single fixed practice-oriented row
                # used to appear after every answer regardless.
                last_meta = _get_meta(last_index)
                st.caption("What next?")
                selected_action = ui.render_action_row(
                    ui.follow_ups_for(
                        last_meta.get("route", ""),
                        abstained=last_meta.get("abstained", False),
                        covered={
                            s.get("route")
                            for s in (last_meta.get("sections") or [])
                        },
                    ),
                    key_prefix="follow_up",
                )

        chat_placeholder = "Ask a question, or attach a screenshot of your work..."
        if pending and pending.get("needs") == "topic":
            chat_placeholder = "Pick a topic above, or type one..."
        elif pending and pending.get("needs") == "subtopic":
            chat_placeholder = "Pick a subtopic above, or type one..."
        elif pending and pending.get("needs") == "attempt":
            chat_placeholder = "Paste your attempt, or attach a screenshot..."

        raw_input = st.chat_input(
            chat_placeholder,
            key="user_query",
            accept_file=True,
            file_type=["png", "jpg", "jpeg", "pdf", "txt", "csv"],
            submit_mode="stop",
        )
        st.caption("Do not include personal information. This tutor can make mistakes.")

    typed_query, uploaded_files = _parse_chat_input(raw_input)
    user_query = ""
    display_user_text = ""

    # 1) Chip click - quick action before the conversation starts, or a
    #    route-aware follow-up after it.
    if selected_action is not None:
        _cancel_pending()
        composed, label, did_clarify = _resolve_action(selected_action)
        if did_clarify:
            st.rerun()
        user_query = composed
        display_user_text = label

    # 2) Completing (or abandoning) a pending clarifying turn
    elif pending is not None:
        selected_value = None
        if pending.get("needs") == "topic":
            selected_value = selected_topic
        elif pending.get("needs") == "subtopic":
            selected_value = selected_subtopic

        composed, escaped, did_clarify_again = _resolve_pending_intent(
            pending, selected_value, typed_query, uploaded_files
        )
        if did_clarify_again:
            st.rerun()
        if escaped:
            _cancel_pending()
            user_query = composed
            display_user_text = composed
        elif composed:
            detail_text = selected_value or typed_query
            if pending.get("needs") == "subtopic":
                detail_text = chains.format_topic_focus(
                    pending.get("parent_topic", ""),
                    detail_text,
                )
            display_user_text = detail_text if detail_text else "Attached attempt for review"
            user_query = composed
            _cancel_pending()
            if detail_text:
                st.session_state.last_practice_topic = detail_text

    # 3) Normal free-form chat, or a starter prompt from the empty state
    else:
        user_query = typed_query
        display_user_text = typed_query
        if not user_query and starter_choice:
            user_query = starter_choice
            display_user_text = starter_choice
            st.session_state.pop("starter_prompt_pills", None)
        if uploaded_files and not user_query:
            user_query = "Please review the attached file(s) and help me with the next step."
            display_user_text = user_query

    if not user_query:
        return

    # ----------------------------------------------------------------------
    # Run the turn
    # ----------------------------------------------------------------------
    attachment_context, unreadable_names, images = extract_attachments(uploaded_files)
    attachment_note = ""
    readable_names = [
        f.name for f in uploaded_files if f.name not in set(unreadable_names)
    ]
    if readable_names:
        attachment_note = f"\n\n[Student attached: {', '.join(readable_names)}]"

    # The router reads this line instead of the pixels; the images themselves
    # go round the agent loop and straight to the chain (see build_ta_agent).
    query_for_model = (
        f"{user_query}{attachment_note}{describe_images(images)}{attachment_context}"
    )
    learning_profile = chains.build_learning_profile(
        query=query_for_model,
        response_mode=sidebar_settings["response_mode"],
        chat_history=st.session_state.chat_history,
    )

    update_session_stats()

    with st.chat_message("Human"):
        ui.md(display_user_text or user_query)
        for uploaded in uploaded_files:
            mime = (uploaded.type or "").lower()
            if mime.startswith("image/"):
                st.image(uploaded, caption=uploaded.name, width="stretch")
            else:
                st.caption(f"Attached: {uploaded.name}")

    route_label = "agent_direct"
    retrieval_debug = []
    tools_used = []
    abstained = False
    retrieval_quality = ""
    diagnostics = {}
    turn_result = {}
    # One entry per answer section. Empty on the fallback path, where a single
    # ungrounded answer is all there is.
    sections = []
    effective_response_mode = sidebar_settings["response_mode"]

    # Wall clock for the whole turn, routing included -- what the student
    # actually waited, not the sum of the parts we happened to measure.
    turn_started = time.perf_counter()

    with st.chat_message("AI", avatar=TA_AVATAR):
        # A status line rather than a bare skeleton: routing is a full LLM call
        # before a single token of the answer appears, and "checking the course
        # schedule" is both an honest progress signal and a chance for the
        # student to notice a misroute before reading a wrong answer.
        status = st.status("Reading your question...", type="compact")
        # Every step of the turn reports through here: the router's choice of
        # tool, each search and what it found, then the writing phase.
        progress = ProgressReporter(sink=_status_sink(status))
        try:
            artifacts = TurnArtifacts()
            agent = build_ta_agent(
                agent_llm=agent_llm,
                contents_db=contents_db,
                documents_db=documents_db,
                chains_dict=all_chains,
                chat_history=st.session_state.chat_history,
                response_mode=sidebar_settings["response_mode"],
                artifacts=artifacts,
                course_context=get_course_context(),
                software_context=get_software_context(),
                course_span=get_course_date_span(),
                progress=progress,
                memory_window=sidebar_settings["memory_window"],
                images=images,
            )
            # Routing + hybrid retrieval happen inside run_ta_turn (agent +
            # tools), and report their own progress as they go.
            turn_result = run_ta_turn(
                agent=agent,
                query=query_for_model,
                chat_history=st.session_state.chat_history,
                artifacts=artifacts,
                memory_window=sidebar_settings["memory_window"],
                progress=progress,
            )

            print('-----turn_result: ', turn_result['tool_calls'])

            route_label = turn_result["route_label"]
            answerable = turn_result.get("answerable_steps") or []

            stream_started = time.perf_counter()
            texts = []

            if answerable:
                # One section per tool call, in the order the model asked for
                # them -- which mirrors the order the student asked. Each gets
                # its own badge and its own sources, so a JMP walkthrough and a
                # concept explanation are never merged under one label.
                for position, step in enumerate(answerable):
                    if position:
                        st.space("small")
                    # Each section says which of them is being written, so a
                    # two-tool turn does not look stalled halfway through.
                    section_label = PHASE_WRITING
                    if len(answerable) > 1:
                        badge = ui.route_meta(step.tool_name)["badge"]
                        section_label = f"Writing the {badge.lower()} part"
                    if step.stream_spec is not None:
                        chain = all_chains.get(step.stream_spec.chain_key)
                        print('-----chain: ', step.stream_spec.chain_key)
                        if chain is None:
                            raise ValueError(
                                f"Unknown stream chain: {step.stream_spec.chain_key}"
                            )
                        # The label flips on the first streamed token.
                        text = _stream_answer(
                            chain.stream(step.stream_spec.payload),
                            progress=progress,
                            label=section_label,
                        )
                    else:
                        progress.emit(label=section_label)
                        text = step.static_answer
                        ui.md(text)
                    weak = step.retrieval_quality == "weak"
                    ui.render_provenance(
                        step.tool_name, abstained=step.abstained, weak=weak
                    )
                    ui.render_sources(
                        step.retrieval_debug,
                        key=f"sources_live_{position}",
                        expanded=weak,
                    )
                    if step.abstained:
                        ui.render_unresolved(
                            get_course_links(), key=f"unresolved_live_{position}"
                        )
                    texts.append(text)
                ai_response = "\n\n".join(t for t in texts if t)
            elif turn_result.get("answer"):
                progress.emit(label=PHASE_WRITING)
                ai_response = turn_result["answer"]
                ui.md(ai_response)
            else:
                progress.emit(
                    detail="No tool prepared an answer — answering directly",
                    tools=["agent_direct"],
                )
                ai_response = _stream_answer(
                    class_chain.stream(
                        chains.build_chain_payload(
                            query=query_for_model,
                            chat_history=st.session_state.chat_history,
                            response_mode=effective_response_mode,
                            memory_window=sidebar_settings["memory_window"],
                        )
                    ),
                    progress=progress,
                )

            stream_ms = int((time.perf_counter() - stream_started) * 1000)
            ai_response_for_history = ai_response
            sections = [
                {
                    "text": text,
                    "route": step.tool_name,
                    "sources": step.retrieval_debug,
                    "abstained": step.abstained,
                    "retrieval_quality": step.retrieval_quality,
                }
                for step, text in zip(answerable, texts)
            ]
            tools_used = turn_result["tools_used"]
            retrieval_debug = turn_result.get("retrieval_debug") or []
            abstained = bool(turn_result.get("abstained", False))
            retrieval_quality = turn_result.get("retrieval_quality") or ""
            diagnostics = {
                "tool_calls": turn_result.get("tool_calls") or [],
                "trace": turn_result.get("trace") or [],
                "retrieval_debug": retrieval_debug,
                "router_ms": turn_result.get("router_ms", 0),
                "router_text": turn_result.get("router_text", ""),
                "stream_ms": stream_ms,
                "route_label": route_label,
                "response_mode": effective_response_mode,
                "abstained": abstained,
                "retrieval_quality": retrieval_quality,
            }

            # The turn collapses to one verdict: what was consulted, how much
            # of it, and how long it took. This is also the label the stored
            # status keeps, so the transcript reads the same as the live turn.
            progress.emit(
                label=ui.completion_label(
                    [section["route"] for section in sections] or [route_label],
                    source_count=len(retrieval_debug),
                    seconds=time.perf_counter() - turn_started,
                    abstained=abstained,
                ),
                state="complete",
            )

        except Exception as e:
            print(e)
            # The exception text is for whoever is debugging, not the student:
            # the label already says the lookup failed, and the fallback answer
            # arrives underneath it either way.
            progress.emit(label=PHASE_FAILED, state="error")
            progress.emit(detail=str(e), debug=True)
            route_label = "fallback_class_chain"
            effective_response_mode = "Direct answer"
            diagnostics = {
                "tool_calls": [], "trace": [], "retrieval_debug": [],
                "router_ms": 0, "router_text": "",
                "stream_ms": 0, "route_label": "fallback_class_chain",
                "response_mode": effective_response_mode, "abstained": False,
            }
            try:
                ai_response_for_history = _stream_answer(
                    class_chain.stream(
                        chains.build_chain_payload(
                            query=query_for_model,
                            chat_history=st.session_state.chat_history,
                            response_mode=effective_response_mode,
                            memory_window=sidebar_settings["memory_window"],
                        )
                    ),
                    progress=progress,
                    label="Answering without course materials",
                )
                progress.emit(
                    label=ui.completion_label(
                        ["fallback_class_chain"],
                        seconds=time.perf_counter() - turn_started,
                    ),
                    state="error",
                )
            except Exception as fallback_error:
                # Both the agent and the fallback failed. Say so in the
                # transcript rather than letting a NameError take the app down.
                print(fallback_error)
                abstained = True
                ai_response_for_history = (
                    "I could not reach the tutoring service for that question. "
                    "Please try again in a moment."
                )
                progress.emit(
                    label="Could not reach the tutoring service", state="error"
                )
                ui.md(ai_response_for_history)

    # Said plainly, on the turn it applies to, rather than left for the student
    # to infer from an answer that quietly ignored their screenshot.
    attachment_notice = ""
    if unreadable_names:
        attachment_notice = "I could not read: " + ", ".join(unreadable_names)
        if len(images) >= MAX_IMAGES:
            attachment_notice += ". " + IMAGE_LIMIT_NOTICE

    practice_topic = (turn_result.get("practice_topic") or "") or chains.infer_topic_from_history(
        st.session_state.chat_history + [HumanMessage(query_for_model)]
    )
    if practice_topic:
        st.session_state.last_practice_topic = practice_topic

    # append AI response to chat history
    history_user_text = (display_user_text or user_query) + attachment_note
    st.session_state.chat_history.append(HumanMessage(history_user_text))
    st.session_state.chat_history.append(AIMessage(ai_response_for_history))
    ai_index = len(st.session_state.chat_history) - 1

    # Two different questions, previously answered by one variable.
    #
    # `abstained` is structural: a tool declined, and the text the student sees
    # IS the refusal. That is what may show an orange badge and a recovery panel.
    #
    # `unresolved` is the analytics signal, and is deliberately looser -- a
    # chain can refuse in prose without the tool abstaining. Using the loose
    # version for the badge put "Not found in course materials" under genuine
    # answers that merely contained the phrase "not covered", which the
    # concept_chain prompt explicitly instructs the model to say when a topic is
    # out of scope.
    unresolved = (
        abstained
        or "don't have enough information" in ai_response_for_history.lower()
        or "not covered" in ai_response_for_history.lower()
    )
    event_payload = build_event_payload(
        event_type="query",
        session_id=st.session_state.session_id,
        mode="unified",
        response_mode=effective_response_mode,
        query=user_query,
        route_label=route_label,
        learning_objective=learning_profile["learning_objective"],
        learner_level=learning_profile["learner_level"],
        resolved=not unresolved,
        metadata={
            "tools_used": tools_used,
            "tool_calls": turn_result.get("tool_calls") or [],
            "source_count": len(retrieval_debug),
            "attachment_count": len(uploaded_files),
            "attachment_names": [f.name for f in uploaded_files],
            "attachment_text_chars": len(attachment_context),
            # Count only. The bytes are the student's own work and have no
            # business in the analytics store.
            "attachment_image_count": len(images),
            # Logged so the weak/abstain rate can be tracked per route in the
            # weekly report, and the thresholds retuned against real traffic.
            "retrieval_quality": retrieval_quality,
        },
    )
    interaction_id = str(store_event(collection, event_payload))

    _set_meta(
        ai_index,
        # Sections drive rendering. The flat fields below stay for the
        # single-answer fallback path and for the follow-up chips, which key
        # off the LAST thing the student read.
        sections=sections,
        route=sections[-1]["route"] if sections else route_label,
        sources=retrieval_debug,
        abstained=abstained,
        interaction_id=interaction_id,
        diagnostics=diagnostics,
        attachment_notice=attachment_notice,
        retrieval_quality=retrieval_quality,
        progress=progress.summary(),
    )

    # Recap card every few student turns. Stored on the message rather than
    # written here: everything drawn after this point is wiped by the rerun.
    st.session_state.student_turns += 1
    if st.session_state.student_turns % RECAP_EVERY == 0:
        try:
            recap_text = all_chains["recap_chain"].invoke({
                "chat_history": chains.format_chat_history(
                    st.session_state.chat_history,
                    max_messages=sidebar_settings["memory_window"],
                )
            })
            _set_meta(ai_index, recap=f"**Where you are so far**\n\n{recap_text}")
        except Exception:
            pass

    st.rerun()


if __name__ == '__main__':
    main()
