"""
Presentation layer for the Virtual TA chat.

Everything here answers one of two student-facing questions:

  "Where did this answer come from?"  -> ROUTE_META and render_provenance()
  "What can I do next?"               -> QUICK_ACTIONS and FOLLOW_UPS

The route used to be recorded only in instructor diagnostics; now every
answer carries a badge naming it.

app.py owns turn orchestration and session state; this module owns how a turn
is drawn. Nothing here calls the agent or mutates chat history.
"""

import json

import streamlit as st

import utils.retrieval as retrieval


# --------------------------------------------------------------------------
# Markdown helpers
# --------------------------------------------------------------------------
def escape_md_dollars(text: str) -> str:
    """Escape $ so Streamlit markdown does not treat it as LaTeX.

    Matters constantly here: analytics answers are full of dollar amounts, and
    a single unescaped "$" swallows the rest of the paragraph into math mode.
    """
    if not text:
        return text
    placeholder = "\u0000"
    protected = text.replace("\\$", placeholder)
    return protected.replace("$", "\\$").replace(placeholder, "\\$")


def _escape_md_link_text(text: str) -> str:
    """Make a document title safe as the [label] half of a markdown link.

    An unescaped "]" closes the label early and the remainder of the title
    leaks out as literal text beside a broken link. Canvas titles are free text
    written by the instructor, so this is a matter of when, not whether.
    """
    return (text or "").replace("[", "\\[").replace("]", "\\]")


def md(text: str, container=None):
    """Render markdown with $ signs escaped."""
    target = container if container is not None else st
    target.markdown(escape_md_dollars(text))


def write_stream_md(stream, container=None):
    """Stream markdown tokens into a placeholder, escaping $ as we go."""
    parent = container if container is not None else st
    placeholder = parent.empty()
    chunks = []
    for chunk in stream:
        text = chunk if isinstance(chunk, str) else getattr(chunk, "content", None) or str(chunk)
        chunks.append(text)
        placeholder.markdown(escape_md_dollars("".join(chunks)))
    return "".join(chunks)


# --------------------------------------------------------------------------
# Route provenance
# --------------------------------------------------------------------------
# One entry per agent tool, plus the two non-tool outcomes. `working` is shown
# live while the tools run, `done` on the status line once the turn has
# finished, and `badge` is what stays under the finished answer. All three say
# the same thing in three tenses, from one table, so they cannot drift apart.
#
# The distinction students actually need is grounded vs. not: facts and
# documents come from the instructor's own material, software help comes from
# the model's general knowledge of JMP/Excel. Only the first kind should look
# authoritative, so only the first kind gets a confident colour.
ROUTE_META = {
    "answer_course_facts": {
        "working": "Checking the course schedule and policies",
        "done": "Checked the course schedule",
        "badge": "Course schedule and policies",
        "icon": ":material/event:",
        "color": "blue",
        "help": "Answered from the synced Canvas schedule and the syllabus facts file.",
    },
    "answer_course_documents": {
        "working": "Reading class recaps and assignment instructions",
        "done": "Read the class recaps and assignment briefs",
        "badge": "Class recaps and assignment briefs",
        "icon": ":material/description:",
        "color": "blue",
        "help": "Answered from instructor-written announcements and assignment pages.",
    },
    "answer_concept": {
        "working": "Searching the class materials",
        "done": "Searched the class materials",
        "badge": "Class materials",
        "icon": ":material/menu_book:",
        "color": "blue",
        "help": "Answered from indexed course content.",
    },
    "answer_software": {
        "working": "Working out the JMP / Excel steps",
        "done": "Worked out the JMP / Excel steps",
        "badge": "JMP / Excel guidance",
        "icon": ":material/build:",
        "color": "violet",
        "help": (
            "General knowledge of the software, grounded by this course's versions "
            "and conventions. Menu paths can differ between versions."
        ),
    },
    "generate_practice": {
        "working": "Writing a practice question",
        "done": "Wrote a practice question",
        "badge": "Practice question",
        "icon": ":material/quiz:",
        "color": "green",
        "help": "Generated for you. Not a past exam question.",
    },
    "coach_practice": {
        "working": "Working out where you're stuck",
        "done": "Coached you on the open question",
        "badge": "Practice coaching",
        "icon": ":material/lightbulb:",
        "color": "green",
        "help": (
            "Guidance on the practice question you have open. Stops short of "
            "the answer on purpose."
        ),
    },
    "check_attempt": {
        "working": "Reviewing your attempt",
        "done": "Reviewed your attempt",
        "badge": "Feedback on your attempt",
        "icon": ":material/rate_review:",
        "color": "green",
        "help": "Rubric-style feedback on what you wrote.",
    },
    "agent_direct": {
        "working": "Thinking it through",
        "done": "Answered directly",
        "badge": "General tutoring",
        "icon": ":material/school:",
        "color": "gray",
        "help": "Answered without looking anything up in the course materials.",
    },
    "fallback_class_chain": {
        "working": "Answering directly",
        "done": "Answered without course materials",
        "badge": "Direct tutoring (fallback)",
        "icon": ":material/warning:",
        "color": "orange",
        "help": "Course lookup failed for this turn, so this answer is not grounded in course materials.",
    },
}

DEFAULT_ROUTE = "agent_direct"


def route_meta(route_label: str) -> dict:
    return ROUTE_META.get(route_label or "", ROUTE_META[DEFAULT_ROUTE])


def _join_route_phrases(route_labels, tense: str) -> str:
    """One sentence naming every route in this turn, in the given tense.

    The router can pick two tools in one turn ("how do I run it, and what does
    it mean"), and both are named -- a student who sees only the first phrase
    has no way to tell a two-part answer from a misroute.
    """
    phrases = []
    for label in route_labels or []:
        phrase = route_meta(label)[tense]
        if phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        return ""
    joined = [phrases[0]] + [p[0].lower() + p[1:] for p in phrases[1:]]
    if len(joined) == 1:
        return joined[0]
    if len(joined) == 2:
        return " and ".join(joined)
    return ", ".join(joined[:-1]) + f", and {joined[-1]}"


def working_label(route_labels) -> str:
    """The live "what I'm doing right now" phrase for one or more routes."""
    return _join_route_phrases(route_labels, "working")


def completion_label(
    route_labels,
    *,
    source_count: int = 0,
    seconds: float = 0.0,
    abstained: bool = False,
) -> str:
    """The one line a finished turn collapses to.

    This is what the student reads at a glance after the answer lands, so it is
    a verdict rather than a transcript: what was consulted, how much of it, and
    how long it took. The detail stays one click away inside the container.
    """
    base = _join_route_phrases(route_labels, "done") or "Answered directly"
    if abstained:
        base = "Found nothing in the course materials"
    parts = [base]
    if source_count:
        parts.append(f"{source_count} source{'' if source_count == 1 else 's'}")
    if seconds:
        parts.append(f"{seconds:.1f}s")
    return " · ".join(parts)


def render_provenance(
    route_label: str, *, abstained: bool = False, weak: bool = False
) -> None:
    """One badge under an answer saying where it came from.

    Cheap to render and it is the only signal a student gets that "the final is
    Aug 12" came out of the synced schedule while "click Analyze > Fit Model"
    came out of the model's head.

    Three states, because grounding is not binary. A loose retrieval match
    still produces a fluent, confident-sounding answer -- the student has no
    way to detect it from the prose, so it has to be said.
    """
    if abstained:
        st.badge(
            "Not found in course materials",
            icon=":material/help:",
            color="orange",
        )
        return
    meta = route_meta(route_label)
    if weak:
        st.badge(
            f"{meta['badge']} — loose match",
            icon=":material/priority_high:",
            color="orange",
            help=(
                "The course materials I found were only a partial match for your "
                "question, so treat this answer as a starting point and check the "
                "sources below."
            ),
        )
        return
    st.badge(meta["badge"], icon=meta["icon"], color=meta["color"], help=meta["help"])


def render_sources(rows, *, key: str, weak: bool = False) -> None:
    """Retrieved course material behind an answer, as a compact popover.

    This used to be gated on a tool name that no longer exists
    (`answer_logistics`), so it never rendered at all. Any retrieval-backed
    route shows its sources now.

    A popover rather than an expander so it sits on one row with the
    provenance badge and the rating thumbs (see render_answer_footer). The
    expander was a full-width block, which pushed the footer to three lines
    under every answer. On a loose match the label says so, since a popover
    cannot be opened by default the way the expander was.
    """
    if not rows:
        return
    label = f"Sources ({len(rows)})"
    if weak:
        label += " — check these"
    with st.popover(
        label,
        type="tertiary",
        icon=":material/library_books:",
        key=key,
        help="The course material this answer was drawn from.",
    ):
        for row in rows:
            source = row.get("source") or "Unknown source"
            url = (row.get("url") or "").strip()
            preview = (row.get("preview") or "").strip()
            # Both indexes open the chunk text with the document title, which
            # the line above already shows.
            if source and preview.startswith(source):
                preview = preview[len(source):].strip()
            # Linked when the index knows where the document lives. A student
            # who wants to check the tutor should be one click from the Canvas
            # page, not left to search for a title they were shown.
            if url:
                st.markdown(f"**[{_escape_md_link_text(source)}]({url})**")
            else:
                st.markdown(f"**{source}**")
            if preview:
                st.caption(preview)


def render_answer_footer(
    route_label: str,
    rows,
    *,
    key: str,
    abstained: bool = False,
    weak: bool = False,
    trailing=None,
) -> None:
    """The one-line strip under an answer: badge, sources, and (optionally)
    whatever the caller wants at the end of the row -- the rating thumbs on
    the last section of a turn.

    One horizontal container, vertically centred, so the badge, the sources
    popover and the thumbs read as a single status line rather than three
    stacked blocks of different heights.
    """
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        render_provenance(route_label, abstained=abstained, weak=weak)
        render_sources(rows or [], key=key, weak=weak)
        if trailing is not None:
            trailing()


def render_unresolved(links: dict, *, key: str) -> None:
    """What to do when the tutor could not answer.

    An abstention used to end the turn with a sentence and nothing else. The
    honest answer is still "I don't know", but the student should not have to
    work out where to go next on their own.
    """
    with st.container(border=True, key=key):
        st.markdown("**Where to look instead**")
        bullets = ["Rephrase with the assignment name or class date, and I will search again"]
        if links.get("canvas_url"):
            bullets.append(f"Check the [course Canvas page]({links['canvas_url']})")
        if links.get("instructor_email"):
            bullets.append(
                f"Email your instructor: [{links['instructor_email']}]"
                f"(mailto:{links['instructor_email']})"
            )
        else:
            bullets.append("Ask your instructor or TA")
        st.markdown("\n".join(f"- {b}" for b in bullets))


def render_progress(record: dict) -> None:
    """Redraw a finished turn's status line, collapsed, above the answer.

    Same element the student watched while the answer was being built, in its
    final state -- so "how did it get this?" stays answerable after the fact
    instead of disappearing with the rerun.
    """
    if not record:
        return
    st.status(
        record.get("label") or "Answer complete",
        state=record.get("state") or "complete",
        type="compact",
        expanded=False,
    )


# --------------------------------------------------------------------------
# Action chips
# --------------------------------------------------------------------------
# Two kinds:
#   kind="query"  -> send `value` straight to the agent, show `label` in the
#                    transcript as what the student said
#   kind="intent" -> start a clarifying turn (see app.py's pending_intent)
QUICK_ACTIONS = [
    {
        "label": "Explain a concept",
        "icon": ":material/menu_book:",
        "kind": "intent",
        "value": "explain",
        "needs": "topic",
        "clarify": "What concept would you like to learn? Pick a topic below or type it in the chat.",
    },
    {
        "label": "Practice question",
        "icon": ":material/quiz:",
        "kind": "intent",
        "value": "practice",
        "needs": "topic",
        "clarify": "What would you like to practice? Pick a topic below or type it in the chat.",
    },
    {
        "label": "Check my work",
        "icon": ":material/rate_review:",
        "kind": "intent",
        "value": "check",
        "needs": "attempt",
        "clarify": "Paste your attempt in the chat, or attach a file, and I will check it.",
    },
    {
        "label": "What's due",
        "icon": ":material/event_upcoming:",
        "kind": "query",
        "value": "What assignments are coming up, and when are they due?",
    },
]

# Concrete openers for the empty state, spanning all four grounded routes so
# the first click teaches the student what this tutor actually covers.
#
# Every one of these must be answerable from the current course data. The
# first used to be "When is the midterm exam?" while no midterm date existed
# in schedule.json or facts.toml, so the very first suggested click abstained.
STARTER_PROMPTS = [
    "When are office hours, and where?",
    "What do I need to do for Individual Eastville Part 1 assignment?",
    "What did we cover in class recently?",
    "How do I run a regression in JMP?",
    "What does an R-squared of 0.62 mean?",
]

# Three fixed chips after every answer. They reuse the quick-action intents:
# "Explain a concept" and "Check my work" open the same clarifying turn as the
# empty-state buttons, and "Practice this" drills the topic of the last answer
# (or asks for one when there is none yet). Three, not more: four stretched
# buttons already wrap awkwardly on a phone, and the chat input is always
# there for anything not listed.
FOLLOW_UPS = [
    {
        "label": "Explain a concept",
        "icon": ":material/menu_book:",
        "kind": "intent",
        "value": "explain",
        "needs": "topic",
        "clarify": "Which concept should I explain? Pick a topic below or type it in the chat.",
    },
    {
        "label": "Practice this",
        "icon": ":material/quiz:",
        "kind": "intent",
        "value": "practice_same",
    },
    {
        "label": "Check my work",
        "icon": ":material/rate_review:",
        "kind": "intent",
        "value": "check",
        "needs": "attempt",
        "clarify": "Paste your attempt in the chat, or attach a file, and I will check it.",
    },
]


def render_action_row(actions: list, *, key_prefix: str):
    """Draw a responsive row of action buttons; return the one clicked, or None.

    A horizontal container rather than `st.columns(len(actions))`: columns keep
    their ratios on a narrow screen and squeeze four labels into unreadable
    slivers, which is exactly the width a student on a phone has.
    """
    clicked = None
    with st.container(horizontal=True, horizontal_alignment="distribute"):
        for idx, action in enumerate(actions):
            if st.button(
                action["label"],
                icon=action.get("icon"),
                key=f"{key_prefix}_{idx}",
                width="stretch",
            ):
                clicked = action
    return clicked


# --------------------------------------------------------------------------
# Instructor diagnostics
# --------------------------------------------------------------------------
def render_diagnostics(diag, *, key):
    """Instructor-only view of what the router decided and what each tool did.

    Tool args alone stopped being enough once dates were resolved in Python:
    knowing the router passed on_date='july 30' says nothing about whether that
    became the right ymd range, hit the right filter, or matched any documents.
    The trace records the resolution, so a bad answer can be attributed to the
    router, the date parsing, the filter, or the retrieval.
    """
    if not diag:
        return

    calls = diag.get("tool_calls") or []
    trace = diag.get("trace") or []
    label = ", ".join(c.get("name", "?") for c in calls) or "no tool (answered directly)"

    with st.expander(f"Diagnostics — {label}", expanded=False,
                     icon=":material/bug_report:", key=key):
        cols = st.columns(3)
        cols[0].metric("Router", f"{diag.get('router_ms', 0)} ms")
        cols[1].metric("Stream", f"{diag.get('stream_ms', 0)} ms")
        cols[2].metric("Tools called", len(calls))

        st.markdown("**Router decision**")
        if calls:
            for i, call in enumerate(calls, start=1):
                st.markdown(f"{i}. `{call.get('name', 'unknown')}`")
                st.code(json.dumps(call.get("args") or {}, indent=2,
                                   ensure_ascii=False), language="json")
        else:
            st.caption("The router chose no tool, so its own reply was the answer.")
        if diag.get("router_text"):
            st.caption(f"Router said: {diag['router_text']}")

        if trace:
            st.markdown("**What each tool actually did**")
            for step in trace:
                bits = [f"chain `{step.get('chain') or 'none'}`"]
                bits.append("retrieval on" if step.get("retrieval") else "no retrieval")
                if "hits" in step:
                    bits.append(f"{step['hits']} hit(s)")
                if step.get("quality"):
                    quality = f"quality **{step['quality']}**"
                    # "filter" means an exact metadata match selected the
                    # documents and no distance was consulted, so the absence
                    # of a distance below is expected rather than missing.
                    if step.get("mode") == "filter":
                        quality += " (via date/type filter)"
                    bits.append(quality)
                if step.get("abstained"):
                    bits.append("ABSTAINED")
                st.markdown(f"`{step.get('tool')}` — {' · '.join(bits)}")

                # The numbers behind the abstain/weak decision. Without these a
                # bad call cannot be attributed to the threshold, the embedding,
                # or the index, and the thresholds cannot be retuned from real
                # traffic (see scripts/calibrate_retrieval.py).
                if step.get("best_distance") is not None:
                    st.caption(
                        f"best distance {step['best_distance']} "
                        f"(strong ≤ {retrieval.STRONG_MAX_DISTANCE}, "
                        f"abstain > {retrieval.ABSTAIN_MIN_DISTANCE}) · "
                        f"content overlap {step.get('best_overlap', 0)} · "
                        f"{step.get('query_tokens', 0)} content token(s) in query"
                    )

                resolved = step.get("resolved") or {}
                if any(resolved.values()):
                    st.caption(
                        f"dates resolved to {resolved.get('from_ymd') or '—'}"
                        f" .. {resolved.get('to_ymd') or '—'}"
                        + (f" (since {resolved['since_ymd']})" if resolved.get("since_ymd") else "")
                    )
                if step.get("filter"):
                    st.code(json.dumps(step["filter"], ensure_ascii=False), language="json")
                if step.get("why"):
                    st.caption(step["why"])

        if diag.get("retrieval_debug"):
            st.markdown("**Retrieved chunks**")
            st.dataframe(diag["retrieval_debug"])
        elif any(s.get("retrieval") for s in trace):
            st.caption("Retrieval ran but returned nothing.")
