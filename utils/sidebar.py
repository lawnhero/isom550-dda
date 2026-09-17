import streamlit as st
from datetime import datetime

from utils.course_context import get_course_banner

# Recent-message window is an internal tuning knob, not a student control.
# Keep in sync with chains_lcel.DEFAULT_MEMORY_WINDOW.
DEFAULT_MEMORY_WINDOW = 8

# One switch, not three modes. The old segmented control offered Direct /
# Hint-first / Step-by-step and only two of nine chains read it at all. In
# 508 logged turns the default was used 459 times, Direct 48, Hint-first
# once -- and hint-first on an EXPLANATION was the wrong instinct anyway;
# the tutor withholds answers where that teaches something (the open
# practice question, via coach_practice). The switch keeps the one
# distinction students actually used and maps onto the same two
# `response_mode` values the chains and the analytics already know.
GUIDED_MODE = "Teach me step-by-step"
DIRECT_MODE = "Direct answer"
DEFAULT_RESPONSE_MODE = GUIDED_MODE

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
    st.subheader("Where answers come from")
    st.markdown(
        "Every answer carries a badge naming its source, so you can tell what is "
        "grounded in course material and what is not.\n\n"
        "- :blue-badge[Course schedule and policies] — the synced Canvas schedule "
        "and the syllabus\n"
        "- :blue-badge[Class recaps and assignment briefs] — announcements and "
        "assignment pages written by your instructor\n"
        "- :blue-badge[Class materials] — the indexed course content\n"
        "- :violet-badge[JMP / Excel guidance] — general knowledge of the software, "
        "grounded by this course's versions and conventions\n"
        "- :green-badge[Practice question] / :green-badge[Practice coaching] / "
        ":green-badge[Feedback on your attempt] — generated for you, not taken "
        "from past exams\n"
        "- :gray-badge[General tutoring] — no course lookup happened\n\n"
        "If the tutor cannot find something, it says so rather than guessing."
    )
    st.subheader("Getting better results")
    st.markdown(
        "- Include context: your dataset, your variables, and what you already tried\n"
        "- Build on previous answers instead of starting over\n"
        "- State your assumptions so the tutor can correct them early\n"
        "- Ask for hints when you want guided practice rather than the answer\n"
        "- Attach a .txt, .csv, or .pdf and I will read it. Screenshots of "
        "your JMP or Excel output work too — I will read the values back to "
        "you first so you can catch a misread"
    )
    st.subheader("Good to know")
    st.markdown(
        "- This tutor is a work in progress and is still being improved\n"
        "- Use it for learning, not for cheating\n"
        "- Never include personal information in your questions"
    )

# Everything a conversation accumulates. Clearing used to reset four of these
# and leave the rest, so a "cleared" chat still carried the previous topic into
# the next practice question and kept old feedback ids alive.
_CONVERSATION_KEYS = (
    "message_meta",
    "pending_intent",
    "last_practice_topic",
    "practice_session",
    "feedback_submitted_ids",
    "clarify_topic_pills",
    "clarify_subtopic_pills",
    "starter_prompt_pills",
)


def clear_chat_history():
    """Clear the chat history and reset every conversation-scoped key."""
    st.session_state.chat_history = []
    for key in _CONVERSATION_KEYS:
        st.session_state.pop(key, None)
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

def _render_outline() -> None:
    """What the tutor can explain, from the same CSV that drives the pills.

    Listed as modules in teaching order with their topics, so the sidebar
    answers "can you help with X?" before the student has to ask -- and so
    the list can never disagree with the pills or the index, because all
    three are derived from the one file.
    """
    from utils.concept_taxonomy import outline

    modules = outline()
    if not modules:
        return
    with st.expander("What I can help with", icon=":material/menu_book:"):
        for module in modules:
            labels = [t["label"] for t in module["topics"]]
            # A module with one topic that repeats its own name has nothing
            # to add on the second line.
            if labels and labels != [module["label"]]:
                topics = " · ".join(labels)
                st.markdown(f"**{module['label']}**  \n<small>{topics}</small>", unsafe_allow_html=True)
            else:
                st.markdown(f"**{module['label']}**")
        st.caption(
            "Plus due dates, class recaps, assignment instructions, and "
            "JMP / Excel steps."
        )


def _render_footer() -> None:
    """Course, term, when the schedule was last synced, and who to email.

    The sync date is the one honest freshness signal the app has: a student
    who sees "synced Aug 21" knows how much to trust a due date without
    reading the reliability advisory the prompt gets.
    """
    banner = get_course_banner()
    head = " · ".join(x for x in (banner.get("code"), banner.get("term")) if x)
    if head:
        st.caption(head)
    if banner.get("synced_on"):
        st.caption(f"Schedule synced {banner['synced_on']}")
    if banner.get("instructor_name"):
        st.caption(banner["instructor_name"])
    if banner.get("instructor_email"):
        st.caption(f"[{banner['instructor_email']}](mailto:{banner['instructor_email']})")


def sidebar():
    """Sidebar: one primary action, two quiet ones, one switch, and context.

    Top to bottom in order of how often a student reaches for it: starting
    over, then saving / help, then the guidance switch, then what the tutor
    covers, then who runs the course.
    """
    with st.sidebar:
        if st.button("New chat", icon=":material/add_comment:", type="primary", width="stretch"):
            clear_chat_history()

        with st.container(horizontal=True):
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

        st.space("small")
        st.session_state.setdefault("guided_steps", True)
        guided = st.toggle(
            "Show me how to work through it",
            key="guided_steps",
            help=(
                "On: concept and assignment answers end with a short ordered "
                "plan for applying them. Off: just the answer and one thing to "
                "check. Deadlines, policies, and JMP / Excel steps are always "
                "answered directly either way."
            ),
        )
        response_mode = GUIDED_MODE if guided else DIRECT_MODE
        # Kept under its old key: the analytics events and the chains still
        # speak in response_mode values.
        st.session_state.response_mode = response_mode

        # Instructor-only controls: hidden unless unlocked via ?debug= in the URL.
        if diagnostics_unlocked():
            # Both need explicit keys. Passing a session-state value as `value`
            # with no key makes Streamlit regenerate the widget id whenever that
            # value changes, which re-creates the widget from its default and
            # silently discards the click -- the diagnostics toggle could not be
            # turned on at all.
            st.session_state.setdefault("memory_window", DEFAULT_MEMORY_WINDOW)
            st.session_state.setdefault("show_diagnostics", False)
            memory_window = st.slider(
                "Recent message window",
                min_value=4,
                max_value=16,
                step=2,
                key="memory_window",
                help="Number of recent messages considered before using summary memory.",
            )
            show_diagnostics = st.toggle(
                "Show agent diagnostics",
                key="show_diagnostics",
                help="Display tool calls and retrieval debug info for development.",
            )
        else:
            # No widgets here, so these have to be written back by hand.
            memory_window = DEFAULT_MEMORY_WINDOW
            show_diagnostics = False
            st.session_state.memory_window = memory_window
            st.session_state.show_diagnostics = show_diagnostics

        st.space("small")
        _render_outline()

        st.space("medium")
        _render_footer()

    return {
        "response_mode": response_mode,
        "memory_window": memory_window,
        "show_diagnostics": show_diagnostics,
    }

