import json
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.globals import set_verbose
import uuid

import utils.chains_lcel as chains
from utils.agent_graph import build_ta_agent, run_ta_turn
from utils.sidebar import sidebar, update_session_stats
import utils.llm_models as llms
from utils.ta_tools import TurnArtifacts, format_source_block_from_debug

# Set the page_title
st.set_page_config(
    page_title="ISOM 550 DDA Virtual TA", page_icon="📚", layout="wide"
)

# cache the vectorized embedding database 
from utils.utils import (
    load_db,
    query_db_connection,
    process_and_store_query,
    build_event_payload,
    store_event,
    store_feedback,
)

# Disable verbose chain debug logs in normal app use.
set_verbose(False)

# 1. Load the Vectorised database
course_path = 'data/course'
contents_path = 'data/contents'
course_db = load_db(db_path=course_path)
contents_db = load_db(db_path=contents_path)

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

CONTINUE_ACTIONS = [
    {
        "label": "Another one on this topic",
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


def _resolve_continue_action(action: dict):
    """Resolve a continue-practice action into a concrete query or clarifying state."""
    intent = action["intent"]
    topic = (st.session_state.get("last_practice_topic") or "").strip()

    if intent == "practice_same":
        if not topic:
            topic = chains.infer_topic_from_history(st.session_state.chat_history)
        if topic:
            return chains.compose_quick_action_query("practice", topic=topic), False
        _append_clarifying_turn(
            action["label"],
            "What topic do you want to practice? Pick one below or type it in the chat.",
            "practice",
            "topic",
        )
        return "", True

    if intent == "practice_harder":
        if not topic:
            topic = chains.infer_topic_from_history(st.session_state.chat_history)
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
        st.session_state.show_continue_actions = False
        _append_clarifying_turn(
            action["label"],
            "Which topic would you like to switch to? Pick one below or type it in the chat.",
            "practice",
            "topic",
        )
        return "", True

    return "", False


def _resolve_pending_intent(pending, typed_query, uploaded_files):
    """Compose a concrete query once the student supplies the missing detail."""
    intent = pending["intent"]
    needs = pending["needs"]

    if needs == "topic":
        topic = (typed_query or "").strip()
        if not topic:
            return ""
        return chains.compose_quick_action_query(intent, topic=topic)

    if needs == "attempt":
        if not typed_query and not uploaded_files:
            return ""
        return chains.compose_quick_action_query(intent, attempt_text=typed_query)

    return typed_query


def _should_show_continue_actions(tools_used):
    return any(tool in {"generate_practice", "check_attempt"} for tool in tools_used)


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
    if "memory_summary" not in st.session_state:
        st.session_state.memory_summary = ""
    if "last_interaction_id" not in st.session_state:
        st.session_state.last_interaction_id = ""
    if "recap_count" not in st.session_state:
        st.session_state.recap_count = 0
    if "feedback_submitted_ids" not in st.session_state:
        st.session_state.feedback_submitted_ids = []
    if "pending_intent" not in st.session_state:
        st.session_state.pending_intent = None
    if "show_continue_actions" not in st.session_state:
        st.session_state.show_continue_actions = False
    if "last_practice_topic" not in st.session_state:
        st.session_state.last_practice_topic = ""
    if "last_tool_calls" not in st.session_state:
        st.session_state.last_tool_calls = []
    if "last_retrieval_debug" not in st.session_state:
        st.session_state.last_retrieval_debug = []

    st.header("Virtual TA - ISOM 550 DDA")
    sidebar_settings = sidebar()
    initial_text = (
        "I can answer logistics questions, explain analytics concepts, and guide you "
        "step-by-step based on your selected response style."
    )
    st.write(
        "Ask any ISOM 550 question. I choose the right tutoring action for your request, "
        "then adapt the explanation to your selected response style."
    )
    option = "unified"
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = [AIMessage(initial_text)]

    # Display conversation statistics in main area
    if len(st.session_state.chat_history) > 1:
        st.caption(f"Conversation: {len(st.session_state.chat_history)} messages")

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                _md(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🦜"):
                _md(message.content)

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

    # Continue-practice buttons after a practice/check turn.
    selected_continue_action = None
    if st.session_state.show_continue_actions and not st.session_state.pending_intent:
        st.caption("Keep practicing")
        continue_cols = st.columns(len(CONTINUE_ACTIONS))
        for idx, action in enumerate(CONTINUE_ACTIONS):
            if continue_cols[idx].button(
                action["label"],
                key=f"continue_action_{idx}",
                width='stretch',
            ):
                selected_continue_action = action

    # Topic pills while waiting for a clarifying detail.
    selected_topic = None
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

    # Pin composer controls so students always see them while scrolling history.
    selected_action = None
    with st.bottom:
        st.caption("Quick actions")
        qa_cols = st.columns(len(QUICK_ACTIONS))
        for idx, action in enumerate(QUICK_ACTIONS):
            if qa_cols[idx].button(action["label"], key=f"quick_action_{idx}", width='stretch'):
                selected_action = action

        chat_placeholder = "Ask a question, or paste a screenshot of your work..."
        if pending and pending.get("needs") == "topic":
            chat_placeholder = "Type the topic you want to focus on..."
        elif pending and pending.get("needs") == "attempt":
            chat_placeholder = "Paste your attempt, or attach a file..."

        raw_input = st.chat_input(
            chat_placeholder,
            key="user_query",
            accept_file="multiple",
            file_type=["png", "jpg", "jpeg", "pdf", "txt", "csv"],
            submit_mode="stop",
        )

    typed_query, uploaded_files = _parse_chat_input(raw_input)
    user_query = ""
    display_user_text = ""

    # 1) Continue-practice action click
    if selected_continue_action is not None:
        composed, did_clarify = _resolve_continue_action(selected_continue_action)
        if did_clarify:
            st.rerun()
        user_query = composed
        display_user_text = selected_continue_action["label"]

    # 2) Fresh quick action click
    elif selected_action is not None:
        st.session_state.pending_intent = None
        st.session_state.show_continue_actions = False
        composed, did_clarify = _resolve_quick_action(
            selected_action, st.session_state.chat_history
        )
        if did_clarify:
            st.rerun()
        user_query = composed
        inferred_topic = chains.infer_topic_from_history(st.session_state.chat_history)
        if inferred_topic:
            display_user_text = f"{selected_action['label']} — {inferred_topic}"
        else:
            display_user_text = selected_action["label"]

    # 3) Completing a pending clarifying turn
    elif pending is not None:
        detail_text = selected_topic or typed_query
        composed = _resolve_pending_intent(pending, detail_text, uploaded_files)
        if composed:
            display_user_text = detail_text if detail_text else "Attached attempt for review"
            user_query = composed
            st.session_state.pending_intent = None
            st.session_state.pop("clarify_topic_pills", None)

    # 4) Normal free-form chat
    else:
        user_query = typed_query
        display_user_text = typed_query
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
                    chains_dict=all_chains,
                    chat_history=st.session_state.chat_history,
                    memory_summary=st.session_state.memory_summary,
                    response_mode=sidebar_settings["response_mode"],
                    artifacts=artifacts,
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
                                memory_summary=st.session_state.memory_summary,
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
                    source_block = format_source_block_from_debug(retrieval_debug)
                    st.markdown("**Sources**")
                    _md(source_block)
                    ai_response_for_history = f"{ai_response}\n\n**Sources**\n{source_block}"

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
                        "memory_summary": st.session_state.memory_summary,
                        "response_mode": effective_response_mode,
                        "learning_objective": learning_profile["learning_objective"],
                        "learner_level": learning_profile["learner_level"],
                        "attempt_check": learning_profile["attempt_check"],
                    }))

        if _should_show_continue_actions(tools_used):
            st.session_state.show_continue_actions = True
            practice_topic = turn_result.get("practice_topic") or chains.infer_topic_from_history(
                st.session_state.chat_history
            )
            if practice_topic:
                st.session_state.last_practice_topic = practice_topic
        else:
            st.session_state.show_continue_actions = False

        # append AI response to chat history
        history_user_text = (display_user_text or user_query) + attachment_note
        st.session_state.chat_history.append(HumanMessage(history_user_text))
        st.session_state.chat_history.append(AIMessage(ai_response_for_history))

        # Save legacy and normalized events.
        process_and_store_query(collection, query=user_query)
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

        # Update memory summary with recent turns.
        recent_turns = chains.format_chat_history(
            st.session_state.chat_history,
            max_messages=sidebar_settings["memory_window"],
        )
        try:
            st.session_state.memory_summary = all_chains["summary_chain"].invoke(
                {
                    "previous_summary": st.session_state.memory_summary or "No summary yet.",
                    "recent_turns": recent_turns,
                }
            )
        except Exception:
            pass

        # Recap card every 4 user turns.
        st.session_state.recap_count += 1
        if st.session_state.recap_count % 4 == 0:
            try:
                recap_text = all_chains["recap_chain"].invoke(
                    {
                        "memory_summary": st.session_state.memory_summary,
                        "chat_history": recent_turns,
                    }
                )
                st.info(f"Learning recap:\n\n{recap_text}")
            except Exception:
                pass

        st.rerun()

if __name__ == '__main__':
    main()
