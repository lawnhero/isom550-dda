import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.globals import set_verbose
import uuid

import utils.chains_lcel as chains
from utils.sidebar import sidebar, update_session_stats
import utils.llm_models as llms
from utils.retrieval import hybrid_retrieve, retrieval_debug_rows

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
retriever_course = course_db.as_retriever()
retriever_contents = contents_db.as_retriever() 

# 2. MongoDB Atlas connection
mongo_db = query_db_connection()
collection = mongo_db['ISOM 550']

# 3. Setup LLM and chains
claude_sonnet = llms.claude_sonnet_with_fallback
claude_haiku = llms.claude_haiku_with_fallback
router_llm = llms.openai_gpt4o_mini  # Use JSON-capable model for routing

# Initialize all chains including the routing chain with specialized router LLM
all_chains = chains.get_all_chains(claude_sonnet, claude_haiku, retriever_course, retriever_contents, router_llm)
router = all_chains['router']
chain_dict = all_chains['chain_dict']

# Individual chains
class_chain = all_chains['class_chain']
        
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

    st.header("Virtual TA - ISOM 550 DDA")
    sidebar_settings = sidebar()
    initial_text = (
        "I can answer logistics questions, explain analytics concepts, and guide you "
        "step-by-step based on your selected response style."
    )
    st.write(
        "Ask any ISOM 550 question. I auto-route between course logistics and learning guidance, "
        "then adapt the explanation to your selected response style."
    )
    option = "unified"
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state or not st.session_state.chat_history:
        st.session_state.chat_history = [AIMessage(initial_text)]

    # Display conversation statistics in main area
    if len(st.session_state.chat_history) > 1:
        st.caption(f"💬 Conversation: {len(st.session_state.chat_history)} messages")

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🦜"):
                st.markdown(message.content)

    # Inline feedback controls for latest response.
    if st.session_state.last_interaction_id:
        latest_id = st.session_state.last_interaction_id
        st.caption("Rate the latest response")
        if latest_id in st.session_state.feedback_submitted_ids:
            st.success("Feedback received. Thank you.")
        else:
            up_col, down_col, _spacer = st.columns([1, 1, 8])
            if up_col.button("👍", key=f"thumb_up_{latest_id}", use_container_width=True):
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
            if down_col.button("👎", key=f"thumb_down_{latest_id}", use_container_width=True):
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

    # Quick actions to reduce prompt-writing friction.
    st.caption("Quick actions")
    quick_actions = [
        ("Explain concept", "Explain this concept clearly for ISOM 550 with a simple business example."),
        ("Practice question", "Create one practice question and then guide me with hints."),
        ("Check my attempt", "I tried a solution. Please check my attempt and tell me what to fix next."),
        ("What is next step?", "What is my immediate next step?"),
    ]
    quick_action_query = ""
    qa_cols = st.columns(len(quick_actions))
    for idx, (label, template_query) in enumerate(quick_actions):
        if qa_cols[idx].button(label, key=f"quick_action_{idx}", use_container_width=True):
            quick_action_query = template_query

    # get user query
    typed_query = st.chat_input("Enter your query here...", key="user_query")
    user_query = typed_query or quick_action_query
    if user_query:
        learning_profile = chains.build_learning_profile(
            query=user_query,
            response_mode=sidebar_settings["response_mode"],
            chat_history=st.session_state.chat_history,
        )

        # Update session statistics
        update_session_stats()
        
        # display user query
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Generate AI response based on selected option
        route_label = "contents"
        retrieval_docs = []
        rewritten_query = user_query
        effective_response_mode = sidebar_settings["response_mode"]
        with st.chat_message("AI", avatar="🦜"):
            try:
                history_text = chains.format_chat_history(
                    st.session_state.chat_history,
                    max_messages=sidebar_settings["memory_window"],
                )
                tool_choice = router.invoke({
                    "query": user_query,
                    "chat_history": history_text
                })

                route_label = tool_choice.label if hasattr(tool_choice, "label") else "contents"
                rewritten_query = tool_choice.query if hasattr(tool_choice, "query") else user_query

                if route_label == "course":
                    retrieval_docs = hybrid_retrieve(course_db, rewritten_query, top_k=4)
                    effective_response_mode = "Direct answer"
                else:
                    retrieval_docs = hybrid_retrieve(contents_db, rewritten_query, top_k=4)
                    effective_response_mode = sidebar_settings["response_mode"]

                # Execute the appropriate chain based on routing decision
                response_stream = chains.call_function(
                    tool_name=route_label,
                    query=rewritten_query,
                    chains_dict=chain_dict,
                    chat_history=st.session_state.chat_history,
                    memory_summary=st.session_state.memory_summary,
                    response_mode=effective_response_mode,
                )
                
                # Stream the response
                ai_response = st.write_stream(response_stream)

                if sidebar_settings["show_diagnostics"]:
                    st.caption(f"Router label: `{route_label}`")
                    st.caption(f"Rewritten query: `{rewritten_query}`")
                    st.caption(f"Effective response style: `{effective_response_mode}`")
                    debug_rows = retrieval_debug_rows(retrieval_docs)
                    if debug_rows:
                        st.dataframe(debug_rows, use_container_width=True)

            except Exception as e:
                print(e)
                route_label = "fallback_class_chain"
                effective_response_mode = "Direct answer"
                st.warning("I hit a routing issue and switched to direct tutoring mode for this turn.")
                ai_response = st.write_stream(
                    class_chain.stream({
                        'query': user_query, 
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

        # append AI response to chat history
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(ai_response))

        # Save legacy and normalized events.
        process_and_store_query(collection, query=user_query)
        unresolved = "don't have enough information" in ai_response.lower() or "not covered" in ai_response.lower()
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
                "rewritten_query": rewritten_query,
                "source_count": len(retrieval_docs),
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

if __name__ == '__main__':
    main()
