"""The held practice question: written by app.py, read by three tools."""

from utils import practice


def test_lifecycle_counts_hints_and_attempts():
    session = practice.start("Q: compute the mean", "Descriptive statistics", "harder")
    assert practice.is_active(session)
    assert session["difficulty"] == "harder"
    practice.record_hint(session)
    practice.record_hint(session)
    practice.record_attempt(session)
    assert session["hints_given"] == 2 and session["attempts"] == 1
    block = practice.prompt_block(session)
    assert "Q: compute the mean" in block
    assert "2 hint(s)" in block and "1 attempt(s)" in block


def test_unknown_difficulty_falls_back_to_same():
    assert practice.start("q", "t", "brutal")["difficulty"] == "same"


def test_inactive_session_is_inert():
    assert not practice.is_active(None)
    assert not practice.is_active({"question": "   "})
    assert practice.record_hint(None) is None
    assert practice.prompt_block(None) == ""
    assert practice.question_of(None) == ""


def test_hint_escalates_to_worked_step_after_max_hints():
    session = practice.start("q", "t")
    for _ in range(practice.MAX_HINTS):
        assert practice.effective_request(session, "hint") == "hint"
        practice.record_hint(session)
    assert practice.hints_exhausted(session)
    assert practice.effective_request(session, "hint") == "worked_step"
    # Only bare hints escalate; an explicit clarify request is honoured.
    assert practice.effective_request(session, "clarify") == "clarify"
    assert practice.effective_request(session, "nonsense") == "worked_step"
