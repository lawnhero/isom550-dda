"""Date words from the student's sentence -> exact Chroma filters.

The router copies "july 30" through verbatim; everything calendar-shaped
happens here, in Python, against the course's own date span."""

import pytest

from utils.retrieval import (
    build_document_filter,
    content_overlap,
    parse_course_date,
    resolve_date_range,
)

SPAN = (20260806, 20261123)  # Fall 2026


@pytest.mark.parametrize(
    "text, expected",
    [
        ("july 30", 20260730),
        ("Jul 30", 20260730),
        ("7/30", 20260730),
        ("30 July", 20260730),
        ("2026-09-02", 20260902),
        ("sep 2 2026", 20260902),
        ("monday", 0),
        ("", 0),
        ("13/45", 0),
    ],
)
def test_parse_course_date(text, expected):
    assert parse_course_date(text, SPAN) == expected


def test_year_is_resolved_from_the_course_span():
    # A term crossing New Year: only one candidate year lands inside the span.
    winter = (20261201, 20270315)
    assert parse_course_date("jan 10", winter) == 20270110
    assert parse_course_date("dec 10", winter) == 20261210


def test_resolve_week_is_the_iso_week_containing_the_date():
    # July 30, 2026 is a Thursday -> Mon Jul 27 .. Sun Aug 2.
    assert resolve_date_range("july 30", SPAN, "week") == (20260727, 20260802)
    assert resolve_date_range("july 30", SPAN, "month") == (20260701, 20260731)
    assert resolve_date_range("july 30", SPAN, "day") == (20260730, 20260730)
    assert resolve_date_range("july 30", SPAN, "bogus") == (20260730, 20260730)
    assert resolve_date_range("nonsense", SPAN, "week") == (0, 0)


def test_document_filter_shapes():
    assert build_document_filter() is None
    assert build_document_filter(doc_type="assignment") == {"doc_type": {"$eq": "assignment"}}
    assert build_document_filter(from_ymd=20260730, to_ymd=20260730) == {"ymd": {"$eq": 20260730}}
    ranged = build_document_filter("announcement", since_ymd=20260101, from_ymd=20260727, to_ymd=20260802)
    # A named range overrides the recency window.
    assert ranged == {"$and": [
        {"doc_type": {"$eq": "announcement"}},
        {"ymd": {"$gte": 20260727}},
        {"ymd": {"$lte": 20260802}},
    ]}
    assert build_document_filter(since_ymd=20260801) == {"ymd": {"$gte": 20260801}}


def test_content_overlap_ignores_stopwords():
    assert content_overlap("What is the capital of Mongolia?", "the capital is Ulaanbaatar") == 0.5
    assert content_overlap("what is the", "anything") == 0.0
