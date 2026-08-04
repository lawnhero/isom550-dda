import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.globals import set_verbose
import uuid

import utils.chains_lcel as chains
from utils.agent_graph import build_ta_agent, run_ta_turn
from utils.course_context import get_course_context, get_software_context
from utils.sidebar import sidebar, update_session_stats
import utils.llm_models as llms
from utils.ta_tools import TurnArtifacts

# Set the page_title
st.set_page_config(
    page_title="ISOM 550 DDA Virtual TA", page_icon="📚", layout="wide"
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
course_path = 'data/course'
contents_path = 'data/contents'
documents_path = 'data/tier_c'
course_db = load_db(db_path=course_path)
contents_db = load_db(db_path=contents_path)
# Tier C: class recaps + assignment briefs, built by scripts/build_tier_c.py.
# Missing until that runs; searches then return nothing and the tool abstains.
documents_db = load_db(db_path=documents_path)

# 2. MongoDB Atlas connection
mongo_db = query_db_connection()
collection = mongo_db['ISOM 550']

# 3. Setup LLM and chains
main_tutor = llms.grok_with_sonnet_fallback
claude_haiku = llms.claude_haiku_with_fallback
agent_llm = llms.openai_gpt4o_mini

all_chains = chains.get_all_chains(main_tutor, claude_haiku)
class_chain = all_chains['class_chain']

QUICK_ACTIONS = [
    {
        "label": "Explain concept",
        "intent": "explain",
        "needs": "topic",
        "clarify": (
            "What concept would you like to learn? Pick a topic below or type it in the chat."
        ),
    },
    {
        "label": "Practice question",
        "intent": "practice",
        "needs": "topic",
        "clarify": (
            "What would you like to practice? Pick a topic below or type it in the chat."
        ),
    },
    {
        "label": "Check my attempt",
        "intent": "check",
        "needs": "attempt",
        "clarify": (
            "Paste your attempt in the chat, or attach a screenshot/file, and I'll check it."
        ),
    },
    {
        "label": "What is next step?",
        "intent": "next_step",
        "needs": None,
        "clarify": "",
    },
]

# Concrete openers for students who do not yet know what to ask. Shown in the
# empty main area before the first question, where they cost no extra space.
STARTER_PROMPTS = [
    "When is the midterm exam?",
    "What's the grading policy?",
    "Help with Assignment 3",
    "Guide me through regression analysis",
    "How do I interpret this statistical output?",
]

FOLLOW_UP_ACTIONS = [
    {
        "label": "Practice this topic",
        "intent": "practice_same",
    },
    {
        "label": "Try a harder one",
        "intent": "practice_harder",
    },
    {
        "label": "Switch topic",
        "intent": "switch_topic",
    },
    {
        "label": "Check my attempt",
        "intent": "check",
    },
]


def _parse_chat_input(raw_input):
    """Normalize st.chat_input return value (str or ChatInputValue)."""
    if raw_input is None:
        return "", []
    if isinstance(raw_input, str):
        return raw_input.strip(), []
    text = (getattr(raw_input, "text", None) or raw_input.get("text") or "").strip()
    files = list(getattr(raw_input, "files", None) or raw_input.get("files") or [])
    return text, files


def _escape_md_dollars(text: str) -> str:
    """Escape $ so Streamlit markdown does not treat it as LaTeX."""
    if not text:
        return text
    # Protect already-escaped dollars, then escape the rest.
    placeholder = "\u0000"
    protected = text.replace("\\$", placeholder)
    return protected.replace("$", "\\$").replace(placeholder, "\\$")


def _md(text: str):
    """Render markdown with $ signs escaped for Streamlit LaTeX."""
    st.markdown(_escape_md_dollars(text))


def _render_retrieval_sources(retrieval_debug: list, *, key: str) -> None:
    """Show retrieved course materials in a collapsed expander."""
    if not retrieval_debug:
        return
    with st.expander(
        f"Sources ({len(retrieval_debug)})",
        expanded=False,
        icon=":material/library_books:",
        key=key,
    ):
        for row in retrieval_debug:
            source = row.get("source") or "Unknown source"
            preview = (row.get("preview") or "").strip()
            st.markdown(f"**{source}**")
            if preview:
                st.caption(preview)


def _write_stream_md(stream):
    """Stream markdown tokens while escaping $ for Streamlit LaTeX."""
    placeholder = st.empty()
    chunks = []
    for chunk in stream:
        text = chunk if isinstance(chunk, str) else getattr(chunk, "content", None) or str(chunk)
        chunks.append(text)
        placeholder.markdown(_escape_md_dollars("".join(chunks)))
    return "".join(chunks)


def _show_thinking_placeholder(container):
    """Show a loading placeholder while routing/retrieval runs."""
    try:
        container.skeleton(height=72)
    except Exception:
        container.markdown(":shimmer[Looking up course materials...]")


def _append_clarifying_turn(action_label: str, clarify_text: str, intent: str, needs: str):
    """Record a clarifying turn and wait for the missing topic/attempt."""
    with st.chat_message("Human"):
        _md(action_label)
    with st.chat_message("AI", avatar="🦜"):
        _md(clarify_text)
    st.session_state.chat_history.append(HumanMessage(action_label))
    st.session_state.chat_history.append(AIMessage(clarify_text))
    st.session_state.pending_intent = {
        "intent": intent,
        "label": action_label,
        "needs": needs,
    }


def _resolve_quick_action(action: dict, chat_history):
    """
    Resolve a fresh quick action.

    Fresh clicks that need topic/attempt always ask first (do not infer from history).
    Returns (user_query, did_clarify).
    """
    intent = action["intent"]
    needs = action["needs"]

    if needs is None:
        return chains.compose_quick_action_query(intent), False

    if needs in {"topic", "attempt"}:
        _append_clarifying_turn(action["label"], action["clarify"], intent, needs)
        return "", True

    return "", False


def _conversation_started(chat_history) -> bool:
    """True once the student has sent at least one message."""
    return any("Human" in str(type(message)) for message in (chat_history or []))


def _resolve_follow_up_action(action: dict):
    """Resolve a post-conversation follow-up into a query or clarifying turn."""
    intent = action["intent"]
    topic = (st.session_state.get("last_practice_topic") or "").strip()
    if not topic:
        topic = chains.infer_topic_from_history(st.session_state.chat_history)

    if intent == "practice_same":
        if topic:
            return chains.compose_quick_action_query("practice", topic=topic), False
        _append_clarifying_turn(
            action["label"],
            "What would you like to practice? Pick a topic below or type it in the chat.",
            "practice",
            "topic",
        )
        return "", True

    if intent == "practice_harder":
        if topic:
            return (
                f"Generate a harder practice question on this ISOM 550 topic: {topic}. "
                "Stay strictly on this topic; do not invent an unrelated scenario."
            ), False
        _append_clarifying_turn(
            action["label"],
            "What topic should the harder practice question focus on?",
            "practice",
            "topic",
        )
        return "", True

    if intent == "switch_topic":
        _append_clarifying_turn(
            action["label"],
            "Which topic would you like to switch to? Pick one below or type it in the chat.",
            "practice",
            "topic",
        )
        return "", True

    if intent == "check":
        _append_clarifying_turn(
            action["label"],
            "Paste your attempt in the chat, or attach a screenshot/file, and I'll check it.",
            "check",
            "attempt",
        )
        return "", True

    return "", False


def _advance_to_subtopic_selection(pending: dict, parent_topic: str):
    """Move from topic selection to subtopic pills for the chosen parent topic."""
    parent_topic = (parent_topic or "").strip()
    clarify = (
        f"Which part of **{parent_topic}** do you want to focus on? "
        "Pick a subtopic below or type it in the chat."
    )
    with st.chat_message("Human"):
        _md(parent_topic)
    with st.chat_message("AI", avatar="🦜"):
        _md(clarify)
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

    Returns (user_query, did_clarify_again).
    did_clarify_again=True means we advanced topic -> subtopic and should rerun.
    """
    intent = pending["intent"]
    needs = pending["needs"]

    if needs == "topic":
        choice = (selected_value or typed_query or "").strip()
        if not choice:
            return "", False

        # Pill selection (or typed exact topic label) with subtopics -> ask subtopic next.
        if choice in chains.CURRICULUM_TOPICS and chains.get_subtopics(choice):
            _advance_to_subtopic_selection(pending, choice)
            return "", True

        # Typed free-form focus, or a topic with no subtopics -> generate now.
        return chains.compose_quick_action_query(intent, topic=choice), False

    if needs == "subtopic":
        choice = (selected_value or typed_query or "").strip()
        if not choice:
            return "", False
        focus = chains.format_topic_focus(pending.get("parent_topic", ""), choice)
        return chains.compose_quick_action_query(intent, topic=focus), False

    if needs == "attempt":
        if not typed_query and not uploaded_files:
            return "", False
        return chains.compose_quick_action_query(intent, attempt_text=typed_query), False

    return typed_query, False


def _render_tool_calls(tool_calls):
    """Show ordered tool calls from the latest agent turn."""
    if not tool_calls:
        st.caption("Tool calls: _(none — agent answered without tools)_")
        return
    with st.expander(f"Tool calls ({len(tool_calls)})", expanded=True):
        for idx, call in enumerate(tool_calls, start=1):
            name = call.get("name", "unknown")
            args = call.get("args") or {}
            st.markdown(f"**{idx}. `{name}`**")
            st.code(json.dumps(args, indent=2, ensure_ascii=False), language="json")


# 4. Build an app with streamlit
def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "last_interaction_id" not in st.session_state:
        st.session_state.last_interaction_id = ""
    if "recap_count" not in st.session_state:
        st.session_state.recap_count = 0
    if "feedback_submitted_ids" not in st.session_state:
        st.session_state.feedback_submitted_ids = []
    if "pending_intent" not in st.session_state:
        st.session_state.pending_intent = None
    if "last_practice_topic" not in st.session_state:
        st.session_state.last_practice_topic = ""
    if "last_tool_calls" not in st.session_state:
        st.session_state.last_tool_calls = []
    if "last_retrieval_debug" not in st.session_state:
        st.session_state.last_retrieval_debug = []
    if "message_sources" not in st.session_state:
        st.session_state.message_sources = {}

    st.header("Virtual TA - ISOM 550 DDA")
    sidebar_settings = sidebar()
    initial_text = (
        "Ask any ISOM 550 question. I can answer logistics questions, explain analytics "
        "concepts, and guide you step-by-step based on your selected response style."
    )
    option = "unified"
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = [AIMessage(initial_text)]

    # Display conversation statistics in main area
    if len(st.session_state.chat_history) > 1:
        st.caption(f"Conversation: {len(st.session_state.chat_history)} messages")

    # display previous conversation history
    for idx, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                _md(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🦜"):
                _md(message.content)
                _render_retrieval_sources(
                    st.session_state.message_sources.get(idx, []),
                    key=f"sources_hist_{idx}",
                )

    # Starter prompts fill the empty main area before the first question.
    starter_choice = None
    if not _conversation_started(st.session_state.chat_history) and not st.session_state.pending_intent:
        st.caption("Try asking")
        starter_choice = st.pills(
            "Example questions",
            options=STARTER_PROMPTS,
            selection_mode="single",
            key="starter_prompt_pills",
            label_visibility="collapsed",
        )

    # Inline feedback controls for latest response.
    if st.session_state.last_interaction_id:
        latest_id = st.session_state.last_interaction_id
        st.caption("Rate the latest response")
        if latest_id in st.session_state.feedback_submitted_ids:
            st.success("Feedback received. Thank you.")
        else:
            up_col, down_col, _spacer = st.columns([1, 1, 8])
            if up_col.button("👍", key=f"thumb_up_{latest_id}", width='stretch'):
                store_feedback(
                    collection=collection,
                    session_id=st.session_state.session_id,
                    interaction_id=latest_id,
                    helpful="Helpful",
                    note="",
                    mode="unified",
                )
                st.session_state.feedback_submitted_ids.append(latest_id)
                st.rerun()
            if down_col.button("👎", key=f"thumb_down_{latest_id}", width='stretch'):
                store_feedback(
                    collection=collection,
                    session_id=st.session_state.session_id,
                    interaction_id=latest_id,
                    helpful="Not helpful",
                    note="",
                    mode="unified",
                )
                st.session_state.feedback_submitted_ids.append(latest_id)
                st.rerun()

    if sidebar_settings["show_diagnostics"] and st.session_state.last_interaction_id:
        _render_tool_calls(st.session_state.last_tool_calls)
        if st.session_state.last_retrieval_debug:
            st.caption("Retrieved chunks")
            st.dataframe(
                st.session_state.last_retrieval_debug,
                width='stretch',
            )
        elif st.session_state.last_tool_calls:
            st.caption("Retrieved chunks: _(none for this tool)_")

    # Topic / subtopic pills while waiting for a clarifying detail.
    selected_topic = None
    selected_subtopic = None
    pending = st.session_state.pending_intent
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

    conversation_started = _conversation_started(st.session_state.chat_history)

    # Pin composer controls so students always see them while scrolling history.
    selected_action = None
    selected_follow_up = None
    with st.bottom:
        if not conversation_started and not pending:
            st.caption("Quick actions")
            qa_cols = st.columns(len(QUICK_ACTIONS))
            for idx, action in enumerate(QUICK_ACTIONS):
                if qa_cols[idx].button(
                    action["label"],
                    key=f"quick_action_{idx}",
                    width="stretch",
                ):
                    selected_action = action
        elif conversation_started and not pending:
            st.caption("What next?")
            follow_cols = st.columns(len(FOLLOW_UP_ACTIONS))
            for idx, action in enumerate(FOLLOW_UP_ACTIONS):
                if follow_cols[idx].button(
                    action["label"],
                    key=f"follow_up_action_{idx}",
                    width="stretch",
                ):
                    selected_follow_up = action

        chat_placeholder = "Ask a question, or paste a screenshot of your work..."
        if pending and pending.get("needs") == "topic":
            chat_placeholder = "Type the topic you want to focus on..."
        elif pending and pending.get("needs") == "subtopic":
            chat_placeholder = "Type the subtopic you want to focus on..."
        elif pending and pending.get("needs") == "attempt":
            chat_placeholder = "Paste your attempt, or attach a file..."

        raw_input = st.chat_input(
            chat_placeholder,
            key="user_query",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "pdf", "txt", "csv"],
            submit_mode="stop",
        )
        st.caption("Do not include personal information. This tutor can make mistakes.")

    typed_query, uploaded_files = _parse_chat_input(raw_input)
    user_query = ""
    display_user_text = ""

    # 1) Follow-up action click (after conversation has started)
    if selected_follow_up is not None:
        composed, did_clarify = _resolve_follow_up_action(selected_follow_up)
        if did_clarify:
            st.rerun()
        user_query = composed
        display_user_text = selected_follow_up["label"]

    # 2) Fresh quick action click (only before conversation starts)
    elif selected_action is not None:
        st.session_state.pending_intent = None
        composed, did_clarify = _resolve_quick_action(
            selected_action, st.session_state.chat_history
        )
        if did_clarify:
            st.rerun()
        user_query = composed
        display_user_text = selected_action["label"]

    # 3) Completing a pending clarifying turn
    elif pending is not None:
        selected_value = None
        if pending.get("needs") == "topic":
            selected_value = selected_topic
        elif pending.get("needs") == "subtopic":
            selected_value = selected_subtopic

        composed, did_clarify_again = _resolve_pending_intent(
            pending, selected_value, typed_query, uploaded_files
        )
        if did_clarify_again:
            st.rerun()
        if composed:
            detail_text = selected_value or typed_query
            if pending.get("needs") == "subtopic":
                detail_text = chains.format_topic_focus(
                    pending.get("parent_topic", ""),
                    detail_text,
                )
            display_user_text = detail_text if detail_text else "Attached attempt for review"
            user_query = composed
            st.session_state.pending_intent = None
            st.session_state.pop("clarify_topic_pills", None)
            st.session_state.pop("clarify_subtopic_pills", None)
            if detail_text:
                st.session_state.last_practice_topic = detail_text

    # 4) Normal free-form chat, or a starter prompt from the empty state
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

    if user_query:
        attachment_note = ""
        if uploaded_files:
            names = ", ".join(f.name for f in uploaded_files)
            attachment_note = f"\n\n[Student attached: {names}]"

        query_for_model = f"{user_query}{attachment_note}"
        learning_profile = chains.build_learning_profile(
            query=query_for_model,
            response_mode=sidebar_settings["response_mode"],
            chat_history=st.session_state.chat_history,
        )

        # Update session statistics
        update_session_stats()
        
        # display user query
        with st.chat_message("Human"):
            _md(display_user_text or user_query)
            for uploaded in uploaded_files:
                mime = (uploaded.type or "").lower()
                if mime.startswith("image/"):
                    st.image(uploaded, caption=uploaded.name, width="stretch")
                else:
                    st.caption(f"Attached: {uploaded.name}")

        route_label = "agent"
        retrieval_debug = []
        tools_used = []
        turn_result = {
            "answer": "",
            "route_label": "agent",
            "tools_used": [],
            "retrieval_debug": [],
            "practice_topic": "",
            "abstained": False,
        }
        effective_response_mode = sidebar_settings["response_mode"]
        with st.chat_message("AI", avatar="🦜"):
            thinking = st.empty()
            try:
                _show_thinking_placeholder(thinking)

                artifacts = TurnArtifacts()
                agent = build_ta_agent(
                    agent_llm=agent_llm,
                    course_db=course_db,
                    contents_db=contents_db,
                    documents_db=documents_db,
                    chains_dict=all_chains,
                    chat_history=st.session_state.chat_history,
                    response_mode=sidebar_settings["response_mode"],
                    artifacts=artifacts,
                    course_context=get_course_context(),
                    software_context=get_software_context(),
                )
                turn_result = run_ta_turn(
                    agent=agent,
                    query=query_for_model,
                    chat_history=st.session_state.chat_history,
                    artifacts=artifacts,
                    memory_window=sidebar_settings["memory_window"],
                )

                thinking.empty()
                stream_spec = turn_result.get("stream_spec")
                if stream_spec is not None:
                    chain = all_chains.get(stream_spec.chain_key)
                    if chain is None:
                        raise ValueError(f"Unknown stream chain: {stream_spec.chain_key}")
                    ai_response = _write_stream_md(chain.stream(stream_spec.payload))
                elif turn_result.get("answer"):
                    ai_response = turn_result["answer"]
                    _md(ai_response)
                else:
                    ai_response = _write_stream_md(
                        class_chain.stream(
                            chains.build_chain_payload(
                                query=query_for_model,
                                chat_history=st.session_state.chat_history,
                                response_mode=effective_response_mode,
                            )
                        )
                    )

                ai_response_for_history = ai_response
                route_label = turn_result["route_label"]
                tools_used = turn_result["tools_used"]
                tool_calls = turn_result.get("tool_calls") or []
                retrieval_debug = turn_result.get("retrieval_debug") or []
                st.session_state.last_tool_calls = tool_calls
                st.session_state.last_retrieval_debug = retrieval_debug

                if "answer_logistics" in tools_used and retrieval_debug:
                    _render_retrieval_sources(
                        retrieval_debug,
                        key="sources_live_current",
                    )

                if sidebar_settings["show_diagnostics"]:
                    _render_tool_calls(tool_calls)
                    st.caption(f"Effective response style: `{effective_response_mode}`")
                    if retrieval_debug:
                        st.caption("Retrieved chunks")
                        st.dataframe(retrieval_debug, width='stretch')
                    else:
                        st.caption("Retrieved chunks: _(none)_")

            except Exception as e:
                print(e)
                thinking.empty()
                route_label = "fallback_class_chain"
                effective_response_mode = "Direct answer"
                st.warning("I hit an agent issue and switched to direct tutoring mode for this turn.")
                st.session_state.last_tool_calls = []
                st.session_state.last_retrieval_debug = []
                ai_response_for_history = _write_stream_md(
                    class_chain.stream({
                        'query': query_for_model, 
                        'chat_history': chains.format_chat_history(
                            st.session_state.chat_history,
                            max_messages=sidebar_settings["memory_window"],
                        ),
                        "response_mode": effective_response_mode,
                        "learning_objective": learning_profile["learning_objective"],
                        "learner_level": learning_profile["learner_level"],
                        "attempt_check": learning_profile["attempt_check"],
                    }))

        practice_topic = turn_result.get("practice_topic") or chains.infer_topic_from_history(
            st.session_state.chat_history + [HumanMessage(query_for_model)]
        )
        if practice_topic:
            st.session_state.last_practice_topic = practice_topic

        # append AI response to chat history
        history_user_text = (display_user_text or user_query) + attachment_note
        st.session_state.chat_history.append(HumanMessage(history_user_text))
        st.session_state.chat_history.append(AIMessage(ai_response_for_history))
        if "answer_logistics" in tools_used and retrieval_debug:
            st.session_state.message_sources[len(st.session_state.chat_history) - 1] = retrieval_debug

        unresolved = (
            turn_result.get("abstained", False)
            or "don't have enough information" in ai_response_for_history.lower()
            or "not covered" in ai_response_for_history.lower()
        )
        event_payload = build_event_payload(
            event_type="query",
            session_id=st.session_state.session_id,
            mode=option,
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
            },
        )
        interaction_id = store_event(collection, event_payload)
        st.session_state.last_interaction_id = str(interaction_id)

        # Recap card every 4 user turns.
        recent_turns = chains.format_chat_history(
            st.session_state.chat_history,
            max_messages=sidebar_settings["memory_window"],
        )
        st.session_state.recap_count += 1
        if st.session_state.recap_count % 4 == 0:
            try:
                recap_text = all_chains["recap_chain"].invoke(
                    {"chat_history": recent_turns}
                )
                st.info(f"Learning recap:\n\n{recap_text}")
            except Exception:
                pass

        st.rerun()

if __name__ == '__main__':
    main()
