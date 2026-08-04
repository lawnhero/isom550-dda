import streamlit as st
from datetime import datetime

# Recent-message window is an internal tuning knob, not a student control.
DEFAULT_MEMORY_WINDOW = 8

def diagnostics_unlocked():
    """
    Agent diagnostics are an instructor tool, not a student-facing control.

    Unlocked by adding ?debug=<value> to the app URL. When a `diagnostics_token`
    secret is configured (deployed app), the value must match it exactly. With no
    secret configured (local dev), any truthy value works.
    """
    supplied = st.query_params.get("debug")
    if not supplied:
        return False
    try:
        expected = st.secrets.get("diagnostics_token")
    except Exception:
        # No secrets.toml present (local dev) - fall back to the simple flag.
        expected = None
    if expected:
        return supplied == expected
    return supplied.lower() in {"1", "true", "yes", "on"}

@st.dialog("How this works", width="large")
def show_help_dialog():
    """Onboarding and policy text, on demand instead of pinned to the sidebar."""
    st.subheader("How questions get answered")
    st.markdown(
        "- Logistics questions (deadlines, grading, policy) are answered from the "
        "syllabus and schedule\n"
        "- Concept and assignment questions are answered from the class materials\n"
        "- If routing fails, the tutor falls back to answering directly"
    )
    st.subheader("Getting better results")
    st.markdown(
        "- Include context: your dataset, your variables, and what you already tried\n"
        "- Build on previous answers instead of starting over\n"
        "- State your assumptions so the tutor can correct them early\n"
        "- Ask for hints when you want guided practice rather than the answer"
    )
    st.subheader("Good to know")
    st.markdown(
        "- This tutor is a work in progress and is still being improved\n"
        "- Use it for learning, not for cheating\n"
        "- Never include personal information in your questions"
    )

def clear_chat_history():
    """Clear the chat history and reset conversation."""
    st.session_state.chat_history = []
    st.session_state.message_sources = {}
    st.session_state.last_interaction_id = ""
    st.session_state.pending_intent = None
    st.rerun()

def save_chat_history():
    """Save chat history in a readable format."""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return "No conversation history to save."
    
    # Create formatted chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_content = f"DDA Virtual TA - Chat History\n"
    chat_content += f"Saved on: {timestamp}\n"
    chat_content += f"Total Messages: {len(st.session_state.chat_history)}\n"
    chat_content += "=" * 50 + "\n\n"
    
    for i, message in enumerate(st.session_state.chat_history, 1):
        if hasattr(message, 'content'):
            if "AI" in str(type(message)) or "Assistant" in str(type(message)):
                chat_content += f"🤖 AI Assistant:\n{message.content}\n\n"
            else:
                chat_content += f"👤 Student:\n{message.content}\n\n"
        chat_content += "-" * 30 + "\n\n"
    
    return chat_content

def sidebar():
    """Enhanced sidebar for MBA Data Analytics Virtual TA."""
    with st.sidebar:
        
        response_modes = ["Direct answer", "Hint-first", "Teach me step-by-step"]
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = "Teach me step-by-step"
        response_mode = st.segmented_control(
            "Response style",
            options=response_modes,
            key="response_mode",
            help="Choose how much guidance the assistant should provide.",
        )
        if response_mode is None:
            response_mode = "Teach me step-by-step"
            st.session_state.response_mode = response_mode

        # Instructor-only controls: hidden unless unlocked via ?debug= in the URL.
        if diagnostics_unlocked():
            memory_window = st.slider(
                "Recent message window",
                min_value=4,
                max_value=16,
                value=st.session_state.get("memory_window", DEFAULT_MEMORY_WINDOW),
                step=2,
                help="Number of recent messages considered before using summary memory.",
            )
            show_diagnostics = st.toggle(
                "Show agent diagnostics",
                value=st.session_state.get("show_diagnostics", False),
                help="Display tool calls and retrieval debug info for development.",
            )
        else:
            memory_window = DEFAULT_MEMORY_WINDOW
            show_diagnostics = False
        st.session_state.memory_window = memory_window
        st.session_state.show_diagnostics = show_diagnostics

        with st.container(horizontal=True):
            if st.button("Clear chat", icon=":material/delete:", width="stretch"):
                clear_chat_history()

            if st.session_state.get("chat_history"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "Save chat",
                    data=save_chat_history(),
                    file_name=f"chat_history_{timestamp}.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    width="stretch",
                )
            else:
                st.button(
                    "Save chat",
                    icon=":material/download:",
                    disabled=True,
                    width="stretch",
                    help="No conversation to save",
                )

        if st.button("How this works", icon=":material/help:", width="stretch"):
            show_help_dialog()

        st.space("medium")
        st.caption("Dr. Wenjun Gu · Goizueta Business School")
        st.caption("wenjun.gu@emory.edu")

    return {
        "response_mode": response_mode,
        "memory_window": memory_window,
        "show_diagnostics": show_diagnostics,
    }

def update_session_stats():
    """Update session statistics (call from main app)."""
    if 'total_queries' in st.session_state:
        st.session_state.total_queries += 1
    else:
        st.session_state.total_queries = 1

def get_sidebar_settings():
    return {
        "response_mode": st.session_state.get("response_mode", "Teach me step-by-step"),
        "memory_window": st.session_state.get("memory_window", DEFAULT_MEMORY_WINDOW),
        "show_diagnostics": st.session_state.get("show_diagnostics", False),
    }
