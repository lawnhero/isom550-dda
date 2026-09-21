"""The practice question currently on the student's screen.

WHY THIS EXISTS

    `check_attempt` used to grade against `topic` plus whatever survived in
    `{chat_history}` -- which `format_chat_history` trims to the last 8
    messages, i.e. four turns. Question -> hint -> clarify -> attempt is
    exactly four turns, so the question could age out of the window precisely
    in the flow a coaching route is meant to encourage. The grader was left
    marking an answer against a topic label.

    Holding the question is therefore a precondition for `coach_practice`, not
    a companion to it: adding hint turns without this would push the question
    out of the window faster and make grading worse.

WHERE IT IS WRITTEN

    Only in app.py, after the stream completes. The question text does not
    exist when `generate_practice` runs -- the tool returns `stream_ready` and
    the text arrives from the stream afterwards -- so the tools READ this and
    app.py WRITES it, the same split `last_practice_topic` already uses. Pure
    dict functions here, no Streamlit import, so this stays testable and so
    chains_lcel can use it without pulling the UI in.

LIFECYCLE

    Replaced when a new question is generated, cleared on chat reset (the key
    is in sidebar._CONVERSATION_KEYS). A graded attempt does NOT clear it: a
    student who gets feedback usually revises, and dropping the question at
    the moment they are most likely to resubmit is the wrong time.
"""

from typing import Any, Dict, Optional

# After this many hints, coaching escalates to a worked step rather than
# handing out a fourth nudge. A counter with no policy at the cap is
# decoration.
MAX_HINTS = 3

DIFFICULTIES = ("easier", "same", "harder")


def start(question: str, topic: str, difficulty: str = "same") -> Dict[str, Any]:
    """A fresh session for a question that has just been shown."""
    return {
        "question": (question or "").strip(),
        "topic": (topic or "").strip(),
        "difficulty": difficulty if difficulty in DIFFICULTIES else "same",
        "hints_given": 0,
        "attempts": 0,
    }


def is_active(session: Optional[Dict[str, Any]]) -> bool:
    """True when there is a question on screen worth coaching against."""
    return bool(session and (session.get("question") or "").strip())


def question_of(session: Optional[Dict[str, Any]]) -> str:
    return (session or {}).get("question", "") or ""


def topic_of(session: Optional[Dict[str, Any]]) -> str:
    return (session or {}).get("topic", "") or ""


def record_hint(session: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not is_active(session):
        return session
    session["hints_given"] = int(session.get("hints_given") or 0) + 1
    return session


def record_attempt(session: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not is_active(session):
        return session
    session["attempts"] = int(session.get("attempts") or 0) + 1
    return session


def hints_exhausted(session: Optional[Dict[str, Any]]) -> bool:
    """True once nudging has stopped working and a worked step is owed."""
    return int((session or {}).get("hints_given") or 0) >= MAX_HINTS


def effective_request(session: Optional[Dict[str, Any]], request: str) -> str:
    """What the coach should actually do, given how much help is already spent.

    The router asks for what the student asked for. This decides whether that
    is still the right move: a fourth `hint` on a question the student has
    been stuck on three times is not help, it is stalling.
    """
    request = (request or "hint").strip().lower()
    if request not in ("hint", "clarify", "worked_step"):
        request = "hint"
    if request == "hint" and hints_exhausted(session):
        return "worked_step"
    return request


def prompt_block(session: Optional[Dict[str, Any]]) -> str:
    """The held question, rendered for a prompt, or "".

    Callers put this in front of a chain instead of hoping the question is
    still inside the history window.
    """
    if not is_active(session):
        return ""
    lines = [f"THE QUESTION ON SCREEN (topic: {topic_of(session) or 'unspecified'}):",
             question_of(session)]
    hints = int(session.get("hints_given") or 0)
    attempts = int(session.get("attempts") or 0)
    if hints or attempts:
        lines.append(
            f"(The student has already had {hints} hint(s) and made "
            f"{attempts} attempt(s) on this question.)"
        )
    return "\n".join(lines)
