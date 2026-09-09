"""Tier A rendering: the facts block the tutor reads on every facts question.

Pure functions, `now` passed explicitly, so the term can be simulated at any
point without touching the real course_data files.
"""

from datetime import datetime, timezone

from utils import course_context as cc

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)

SCHEDULE = {
    "generated_at": "2026-08-30T03:00:00Z",
    "course": {
        "name": "ISOM550-DDA-FA26", "term": "Fall 2026",
        "timezone": "America/New_York", "url": "https://canvas.example/courses/1",
    },
    "modules": [{"position": 1, "name": "M1: Descriptive Analytics"}],
    "assignments": [
        {"name": "Past quiz", "due_utc": "2026-08-26T14:00:00Z",
         "due_local": "Wed Aug 26, 2026, 10:00 AM EDT", "points": 10, "kind": "quiz",
         "url": "https://canvas.example/a/1"},
        {"name": "Next quiz", "due_utc": "2026-09-09T14:00:00Z",
         "due_local": "Wed Sep 9, 2026, 10:00 AM EDT", "points": 7, "kind": "quiz",
         "url": "https://canvas.example/a/2"},
    ],
    "announcements": [
        {"title": "Class 1 (8/24) Intro", "posted_utc": "2026-08-25T01:00:00Z",
         "url": "https://canvas.example/d/1"},
        {"title": "Welcome!", "posted_utc": "2026-08-06T04:00:00Z",
         "url": "https://canvas.example/d/0"},
        # Scheduled for the future: must not appear when rendering at NOW.
        {"title": "Class 9 (10/1) Future", "posted_utc": "2026-10-02T01:00:00Z",
         "url": "https://canvas.example/d/9"},
    ],
    "pages": [{"title": "JMP Regression", "url": "https://canvas.example/p/1"}],
}

FACTS = {
    "term": "Fall 2026",
    "stale_after_days": 12,
    "instructor": {"name": "Prof X", "email": "x@example.edu", "office": "GBS 1",
                   "office_hours": "Wed 1-2 PM"},
    "tas": [{"name": "TBD", "email": ""}],
    "grading": {"weights": "| item | w |"},
    "software": {"jmp": "JMP 18"},
}


def test_render_splits_deadlines_and_hides_unposted_announcements(monkeypatch):
    # The index advisory reads data/documents/provenance.json from the working
    # tree; a fake in-memory schedule has no index to vouch for, and whether
    # that file happens to exist on this machine is not what this test checks.
    monkeypatch.setattr(cc, "_index_advisories", lambda *a, **k: [])
    block = cc.render(SCHEDULE, FACTS, now=NOW)
    assert "UPCOMING DEADLINES" in block and "Next quiz" in block
    assert "PAST DEADLINES" in block and "Past quiz" in block
    assert "Class 1 (8/24) Intro" in block
    assert "Future" not in block
    assert "!! SCHEDULE RELIABILITY" not in block


def test_placeholder_ta_is_not_rendered():
    block = cc.render(SCHEDULE, FACTS, now=NOW)
    assert "TAs:" not in block
    assert "TBD" not in block


def test_real_ta_is_rendered_with_and_without_email():
    facts = {**FACTS, "tas": [{"name": "A. Grader", "email": "a@x.edu"}, {"name": "B. Helper"}]}
    block = cc.render(SCHEDULE, facts, now=NOW)
    assert "TAs: A. Grader (a@x.edu); B. Helper" in block


def test_term_conflict_advisory_leads_the_block():
    facts = {**FACTS, "term": "Spring 2026"}
    block = cc.render(SCHEDULE, facts, now=NOW)
    assert block.startswith("!! SCHEDULE RELIABILITY [conflict]")
    assert "Do NOT state specific due dates" in block


def test_drift_advisory_after_stale_limit():
    # Mid-term (a deadline is still ahead) with a sync older than the limit.
    late = datetime(2026, 9, 5, tzinfo=timezone.utc)
    levels = [a.level for a in cc.advisories(SCHEDULE, FACTS, now=late, stale_after_days=3)]
    assert "drift" in levels and "ended" not in levels


def test_ended_advisory_once_every_deadline_is_past():
    after = datetime(2026, 12, 1, tzinfo=timezone.utc)
    levels = [a.level for a in cc.advisories(SCHEDULE, FACTS, now=after)]
    assert "ended" in levels and "drift" not in levels


def test_software_context_lists_versions_and_walkthroughs():
    block = cc.render_software_context(SCHEDULE, FACTS)
    assert "JMP: JMP 18" in block
    assert "JMP Regression: https://canvas.example/p/1" in block


def test_course_date_span_covers_announcements_and_deadlines():
    assert cc.course_date_span(SCHEDULE) == (20260806, 20261002)
