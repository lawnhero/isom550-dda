import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

import utils.chains_lcel as chains
import utils.ui as ui
from utils.progress import ProgressReporter
from utils.retrieval import (
    RetrievalResult,
    _extract_source_label,
    build_document_filter,
    hybrid_retrieve,
    search_concepts,
    resolve_date_range,
    retrieval_debug_rows,
    search_documents,
)


@dataclass
class StreamSpec:
    """Payload for streaming a tutoring chain in the UI after tool prep."""

    chain_key: str
    payload: Dict[str, Any]


@dataclass
class ToolStep:
    """Everything one tool call prepared, kept apart from every other call."""

    step_id: str
    tool_name: str
    stream_spec: Optional[StreamSpec] = None
    static_answer: str = ""
    sources: List[str] = field(default_factory=list)
    retrieval_debug: List[Dict[str, str]] = field(default_factory=list)
    # "strong" | "weak" | "none" | "" (no retrieval ran). Drives the provenance
    # badge: a loose match is answered, but the student is told it was loose.
    retrieval_quality: str = ""
    abstained: bool = False
    practice_topic: str = ""
    # Per-tool debugging record. Tool args alone are not enough now that dates
    # are resolved in Python: seeing on_date='july 30' tells you nothing about
    # whether it became the right ymd range or hit the right filter.
    trace: Dict[str, Any] = field(default_factory=dict)

    @property
    def produced_answer(self) -> bool:
        return self.stream_spec is not None or bool(self.static_answer)


@dataclass
class TurnArtifacts:
    """Side-channel carrying what each tool prepared, one entry per tool call.

    This used to be a flat set of slots -- one `stream_spec`, one `sources`,
    one `abstained` -- shared by every tool in the turn. The router is told a
    question can need two tools, and LangGraph runs the calls in a single
    parallel batch, so the slots were a race: whichever tool finished LAST won,
    and the other tool's fully prepared answer was dropped on the floor.

    "How do I run a regression in JMP, and what does R-squared mean?" reliably
    called answer_software then answer_concept, and the student got only the
    concept half -- under a badge naming answer_software, because the badge
    read the FIRST tool while the answer came from the LAST.

    Keyed by a per-call `step_id` the tool generates and echoes back in its
    ToolMessage, so concurrent tools cannot collide, and the caller can put the
    steps back into the model's intended order (see run_ta_turn).
    """

    steps: Dict[str, ToolStep] = field(default_factory=dict)

    def new_step(self, tool_name: str) -> ToolStep:
        """Claim a fresh slot for one tool call. Safe to call from any thread."""
        step = ToolStep(step_id=uuid.uuid4().hex, tool_name=tool_name)
        self.steps[step.step_id] = step
        return step

    @property
    def ordered_steps(self) -> List[ToolStep]:
        """Steps in completion order -- a fallback when correlation fails."""
        return list(self.steps.values())

    @property
    def tools_used(self) -> List[str]:
        return [step.tool_name for step in self.steps.values()]


@dataclass
class ToolExecutionResult:
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    retrieval_debug: List[Dict[str, str]] = field(default_factory=list)
    practice_topic: str = ""
    abstained: bool = False
    tool_name: str = ""
    stream_ready: bool = False
    retrieval_quality: str = ""
    # Echoed back through the ToolMessage so run_ta_turn can match this result
    # to the ToolStep that holds its (unserialisable) chain payload.
    step_id: str = ""


def _serialize_tool_result(result: ToolExecutionResult) -> str:
    """The ToolMessage the router sees. Deliberately a receipt, not the answer.

    It used to serialise the whole ToolExecutionResult, including `sources` and
    a `retrieval_debug` array carrying a 120-character preview of every
    retrieved chunk. That put ~250 tokens of real course content into the
    router's context on every retrieval call -- content the router has no use
    for, since the tutoring answer is streamed from the side channel. It also
    gave the post-tool router pass enough material to write its own redundant
    answer to the student's question, which was then discarded.

    What remains is what something actually reads: `step_id` correlates this
    receipt to the ToolStep holding the chain payload, and the rest tells the
    router (and the retry check) whether the call succeeded. Everything else is
    read off the ToolStep instead.
    """
    return json.dumps(
        {
            "answer": result.answer,
            "abstained": result.abstained,
            "tool_name": result.tool_name,
            "stream_ready": result.stream_ready,
            "step_id": result.step_id,
        }
    )


def _extract_source_labels(docs: List[Any]) -> List[str]:
    """Document names for the retrieved chunks.

    Was: render the whole markdown source block, then parse the labels back out
    of it by hunting for "] " and "**:". That round-trip could only survive
    labels with no brackets in them -- and Tier C labels are now Canvas titles,
    which are free text. Ask the labeller directly instead.
    """
    return [_extract_source_label(doc, i) for i, doc in enumerate(docs)]


def _filter_only_question(doc_type: str, days_back: int, on_date: str, date_span: str) -> str:
    """A question for the chain when the router supplied only a date filter.

    Retrieval is happy with an empty query -- the filter picked the documents.
    The tutoring chain is not: `doc_chain`'s prompt ends with "Question: {query}",
    and handed an empty string the model improvises, introducing itself and
    asking what the student wants instead of summarising the eight documents
    sitting directly above it in the prompt.
    """
    kind = {
        "announcement": "class sessions",
        "assignment": "assignments",
    }.get(doc_type, "class documents")

    if on_date:
        window = {
            "week": f"the week of {on_date}",
            "month": f"{on_date}",
        }.get(date_span, f"{on_date}")
        when = f" from {window}"
    elif days_back == 7:
        when = " from the past week"
    elif days_back:
        when = f" from the last {days_back} days"
    else:
        when = ""

    return (
        f"Summarise what these {kind}{when} covered. List each one by name and "
        "say briefly what it was about."
    )


def _in_conversation(chat_history) -> bool:
    """True once the student has said something before this turn.

    Gates the short-query exemption in retrieval.assess: a two-word turn is a
    follow-up mid-conversation and a genuinely thin question as an opener.
    """
    return any("Human" in str(type(m)) for m in (chat_history or []))


ABSTAIN_MESSAGE = (
    "I don't have enough information in the course materials to answer that reliably. "
    "Please check the syllabus or ask your instructor."
)


def _short_query(query: str, limit: int = 70) -> str:
    """The search text, trimmed to fit a status line.

    The router rewrites the student's question before searching and can hand a
    tool two full sentences; unabridged it turns the status log into a wall.
    """
    text = " ".join((query or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_retrieval(
    found: RetrievalResult, what: str, *, query: str = "", window: str = ""
) -> str:
    """The single status line a finished search gets.

    One line, not three. An announcement before the search plus a result after
    it read as two lines only in principle: tools run in ToolNode's workers, so
    both are flushed together and the student sees the same search described
    twice. The list of sources is a third repeat -- that one belongs to the
    Sources expander, which is built for it and can show the text.
    """
    asked = f" for “{_short_query(query)}”" if (query or "").strip() else ""
    if not found.docs:
        return f"Searched {what}{window}{asked} — nothing matched"
    quality = {
        "strong": "close match",
        "weak": "loose match",
        "none": "no usable match",
    }.get(found.quality, found.quality)
    by = " by date/type filter" if found.mode == "filter" else ""
    return (
        f"Searched {what}{window}{asked} — "
        f"{len(found.docs)} passage(s){by}, {quality}"
    )


def _prepare_rag_tool(
    *,
    query: str,
    vector_db,
    chain_key: str,
    chat_history,
    response_mode: str,
    tool_name: str,
    artifacts: TurnArtifacts,
    progress: ProgressReporter,
    module: str = "",
) -> ToolExecutionResult:
    step = artifacts.new_step(tool_name)
    found: RetrievalResult = search_concepts(
        vector_db,
        query,
        module=module,
        top_k=4,
        in_conversation=_in_conversation(chat_history),
    )
    debug_rows = retrieval_debug_rows(found.docs)
    step.retrieval_debug = debug_rows
    progress.emit(detail=_describe_retrieval(found, "the class materials", query=query))

    if found.quality == "none":
        progress.emit(detail="Not enough grounding to answer — abstaining")
        step.abstained = True
        step.static_answer = ABSTAIN_MESSAGE
        step.retrieval_quality = "none"
        step.trace = {
            "tool": tool_name, "chain": None, "retrieval": True,
            "hits": len(found.docs), "abstained": True, **found.as_trace(),
        }
        return ToolExecutionResult(
            answer=ABSTAIN_MESSAGE,
            retrieval_debug=debug_rows,
            abstained=True,
            tool_name=tool_name,
            stream_ready=False,
            retrieval_quality="none",
            step_id=step.step_id,
        )

    payload = chains.build_chain_payload(
        query=query,
        chat_history=chat_history,
        response_mode=response_mode,
        context=chains._format_docs(found.docs),
    )
    sources = _extract_source_labels(found.docs)
    practice_topic = chains.infer_curriculum_topic(query)
    step.sources = sources
    step.practice_topic = practice_topic
    step.stream_spec = StreamSpec(chain_key=chain_key, payload=payload)
    step.retrieval_quality = found.quality
    step.trace = {
        "tool": tool_name, "chain": chain_key, "retrieval": True,
        "hits": len(found.docs), "topic": practice_topic or "(none)",
        "module": (module or "").strip() or "(none)",
        **found.as_trace(),
    }

    return ToolExecutionResult(
        answer="Prepared grounded answer for streaming.",
        sources=sources,
        retrieval_debug=debug_rows,
        practice_topic=practice_topic,
        tool_name=tool_name,
        stream_ready=True,
        retrieval_quality=found.quality,
        step_id=step.step_id,
    )


def build_ta_tools(
    *,
    contents_db,
    documents_db=None,
    chains_dict: Dict[str, Any],
    chat_history,
    response_mode: str,
    artifacts: TurnArtifacts,
    course_context: str = "",
    software_context: str = "",
    course_span=None,
    progress: Optional[ProgressReporter] = None,
):
    """Build agent tools that prepare retrieval/payloads for streamed LCEL answers.

    `progress` receives a line per meaningful step inside each tool. The tools
    run in ToolNode's thread pool, so those lines are buffered and painted at
    the next graph node boundary -- the status LABEL for this phase is set by
    run_ta_turn the moment the router names its tools, which is early enough to
    cover the retrieval that follows.
    """
    progress = progress or ProgressReporter()

    def answer_course_facts(query: str) -> str:
        """Answer questions about dates, people, grading, and what class has covered."""
        step = artifacts.new_step("answer_course_facts")

        if not (course_context or "").strip():
            progress.emit(detail="No course schedule snapshot is loaded — abstaining")
            # No snapshot loaded. Say so rather than let another route invent a date.
            answer = (
                "I don't have the course schedule loaded right now, so I can't confirm "
                "dates or deadlines. Please check Canvas."
            )
            step.static_answer = answer
            step.abstained = True
            return _serialize_tool_result(
                ToolExecutionResult(
                    answer=answer,
                    abstained=True,
                    tool_name="answer_course_facts",
                    stream_ready=False,
                    step_id=step.step_id,
                )
            )

        progress.emit(
            detail=(
                "Reading the synced schedule and syllabus facts "
                f"({len(course_context):,} chars, no search needed)"
            )
        )
        # No retrieval: the Tier A block already is the context.
        payload = {
            "course_context": course_context,
            "chat_history": chains.format_chat_history(chat_history, max_messages=8),
            "query": query,
        }
        step.stream_spec = StreamSpec(chain_key="facts_chain", payload=payload)
        step.trace = {
            "tool": "answer_course_facts", "chain": "facts_chain",
            "retrieval": False, "context_chars": len(course_context),
        }
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared course-facts answer for streaming.",
                tool_name="answer_course_facts",
                stream_ready=True,
                step_id=step.step_id,
            )
        )

    def answer_software(query: str) -> str:
        """Help operate JMP or Excel. Answers from model knowledge, no retrieval."""
        step = artifacts.new_step("answer_software")
        progress.emit(
            detail="Working out the JMP / Excel steps from this course's conventions"
        )

        # Deliberately no vector search. Retrieval here was actively harmful:
        # a JMP question used to land in answer_concept and get four unrelated
        # stats Q&A rows injected as authoritative "course context".
        step.stream_spec = StreamSpec(
            chain_key="software_chain",
            payload={
                "software_context": software_context or "",
                "chat_history": chains.format_chat_history(chat_history, max_messages=8),
                "query": query,
            },
        )
        step.trace = {
            "tool": "answer_software", "chain": "software_chain",
            "retrieval": False, "context_chars": len(software_context or ""),
        }
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared software help for streaming.",
                tool_name="answer_software",
                stream_ready=True,
                step_id=step.step_id,
            )
        )

    def answer_course_documents(
        query: str = "",
        doc_type: str = "",
        days_back: int = 0,
        on_date: str = "",
        date_span: str = "day",
    ) -> str:
        """Look up assignment instructions or what a class covered.

        query:     the topic to search for. MAY BE EMPTY when the question is
                   purely about a date range ("what did we cover last week") --
                   the filter alone then selects the documents.
        doc_type: "assignment", "announcement", or "" for both.
        days_back: restrict to the last N days ("this week" -> 7). 0 = no limit.
        on_date:   a date the student named, copied verbatim from their question
                   ("july 30", "7/30"). No year needed.
        date_span: how much around on_date to cover -- "day" (default), "week"
                   for "the week of July 30", or "month" for "in July".

        `query` used to be required, and the router routinely omitted it on
        exactly the date-range questions this tool exists for -- pydantic
        rejected the call, the tool never ran, and half the student's question
        vanished with no error they could see.
        """
        step = artifacts.new_step("answer_course_documents")

        doc_type = (doc_type or "").strip().lower()
        if doc_type not in {"assignment", "announcement", ""}:
            doc_type = ""

        # Date arithmetic is done here, not by the model. The agent only says
        # how far back to look; turning that into a cutoff is Python's job.
        since_ymd = 0
        try:
            days_back = int(days_back or 0)
        except (TypeError, ValueError):
            days_back = 0
        if days_back > 0:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
            since_ymd = int(cutoff.strftime("%Y%m%d"))

        # A date the student named, plus how wide a window they meant. Parsed
        # here, not by the model: the router copies their words through and
        # Python resolves the year and the calendar boundaries.
        from_ymd, to_ymd = (
            resolve_date_range(on_date, course_span, date_span) if on_date else (0, 0)
        )

        # A week or month can legitimately hold more documents than a single day.
        top_k = 8 if to_ymd > from_ymd else 4

        chroma_filter = build_document_filter(doc_type, since_ymd, from_ymd, to_ymd)
        in_conversation = _in_conversation(chat_history)
        print(
            "[answer_course_documents] search params:",
            {
                "query": (query or "").strip() or "(filter-only)",
                "doc_type": doc_type or "(any)",
                "since_ymd": since_ymd or None,
                "from_ymd": from_ymd or None,
                "to_ymd": to_ymd or None,
                "top_k": top_k,
                "chroma_filter": chroma_filter,
                "in_conversation": in_conversation,
            },
        )

        has_filter = bool(doc_type or since_ymd or (from_ymd and to_ymd))
        if not (query or "").strip() and not has_filter:
            # No topic and nothing to filter on: there is no question here to
            # answer. Say so rather than returning whatever ranks first.
            answer = (
                "I need either a topic or a date to look that up. Tell me the "
                "assignment name or which class you mean."
            )
            step.static_answer = answer
            step.abstained = True
            progress.emit(detail="No topic and no date to search on — asking for one")
            step.trace = {
                "tool": "answer_course_documents", "chain": None,
                "retrieval": False, "why": "no query and no filter",
            }
            return _serialize_tool_result(
                ToolExecutionResult(
                    answer=answer, abstained=True,
                    tool_name="answer_course_documents", stream_ready=False,
                    step_id=step.step_id,
                )
            )

        what = {
            "assignment": "assignment briefs",
            "announcement": "class recaps",
        }.get(doc_type, "class recaps and assignments")
        window = ""
        if from_ymd and to_ymd:
            window = f" dated {from_ymd}–{to_ymd}"
        elif since_ymd:
            window = f" from the last {days_back} days"

        found: RetrievalResult = search_documents(
            documents_db, query, doc_type=doc_type,
            since_ymd=since_ymd, from_ymd=from_ymd, to_ymd=to_ymd, top_k=top_k,
            in_conversation=in_conversation,
        )

        if found.docs:
            print(
                "[answer_course_documents] top hits:",
                [
                    {
                        "title": (getattr(d, "metadata", {}) or {}).get("title"),
                        "doc_type": (getattr(d, "metadata", {}) or {}).get("doc_type"),
                        "ymd": (getattr(d, "metadata", {}) or {}).get("ymd"),
                    }
                    for d in found.docs[:3]
                ],
            )
        debug_rows = retrieval_debug_rows(found.docs)
        step.retrieval_debug = debug_rows
        progress.emit(
            detail=_describe_retrieval(found, what, query=query, window=window)
        )
        if found.quality != "none":
            progress.emit(label=ui.collected_label(["answer_course_documents"]))
        step.trace = {
            "tool": "answer_course_documents", "chain": "doc_chain", "retrieval": True,
            "args": {"doc_type": doc_type or "(any)", "days_back": days_back,
                     "on_date": on_date or "(none)", "date_span": date_span},
            "resolved": {"from_ymd": from_ymd, "to_ymd": to_ymd, "since_ymd": since_ymd},
            "filter": chroma_filter,
            "top_k": top_k, "hits": len(found.docs), **found.as_trace(),
        }

        if found.quality == "none":
            answer = (
                "I couldn't find that in the class recaps or assignment instructions. "
                "Check the Canvas page for the assignment, or ask your instructor."
            )
            step.abstained = True
            step.static_answer = answer
            step.retrieval_quality = "none"
            step.trace["abstained"] = True
            progress.emit(detail="Nothing close enough to answer from — abstaining")
            return _serialize_tool_result(
                ToolExecutionResult(
                    answer=answer,
                    retrieval_debug=debug_rows,
                    abstained=True,
                    tool_name="answer_course_documents",
                    stream_ready=False,
                    retrieval_quality="none",
                    step_id=step.step_id,
                )
            )

        sources = _extract_source_labels(found.docs)
        step.sources = sources
        step.stream_spec = StreamSpec(
            chain_key="doc_chain",
            payload={
                "context": chains._format_docs(found.docs),
                "chat_history": chains.format_chat_history(chat_history, max_messages=8),
                "response_mode": response_mode,
                "query": (query or "").strip() or _filter_only_question(
                    doc_type, days_back, on_date, date_span
                ),
            },
        )
        step.retrieval_quality = found.quality
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared course-document answer for streaming.",
                sources=sources,
                retrieval_debug=debug_rows,
                tool_name="answer_course_documents",
                stream_ready=True,
                retrieval_quality=found.quality,
                step_id=step.step_id,
            )
        )

    def answer_concept(query: str, module: str = "") -> str:
        """Explain analytics concepts, assignment help, coding, or interpretation using class materials."""
        result = _prepare_rag_tool(
            query=query,
            vector_db=contents_db,
            chain_key="step_chain",
            chat_history=chat_history,
            response_mode=response_mode,
            tool_name="answer_concept",
            artifacts=artifacts,
            progress=progress,
            module=module,
        )
        return _serialize_tool_result(result)

    def generate_practice(topic: str, difficulty: str = "same") -> str:
        """Generate one MBA-style practice question for a topic. difficulty: same or harder."""
        topic = (topic or "").strip() or chains.infer_topic_from_history(chat_history) or "General analytics"
        difficulty = (difficulty or "same").strip().lower()
        if difficulty not in {"same", "harder"}:
            difficulty = "same"

        payload = {
            "topic": topic,
            "difficulty": difficulty,
            "learning_objective": chains.infer_learning_objective(topic),
            "learner_level": chains.infer_learner_level(chat_history),
            "chat_history": chains.format_chat_history(chat_history, max_messages=8),
        }
        step = artifacts.new_step("generate_practice")
        progress.emit(
            detail=f"Writing a {difficulty}-difficulty practice question on {topic}"
        )
        step.practice_topic = topic
        step.stream_spec = StreamSpec(chain_key="practice_chain", payload=payload)
        step.trace = {
            "tool": "generate_practice", "chain": "practice_chain",
            "retrieval": False, "topic": topic, "difficulty": difficulty,
        }

        result = ToolExecutionResult(
            answer="Prepared practice question for streaming.",
            practice_topic=topic,
            tool_name="generate_practice",
            stream_ready=True,
            step_id=step.step_id,
        )
        return _serialize_tool_result(result)

    def check_attempt(attempt_text: str, topic: str = "") -> str:
        """Check a student's attempt and return rubric-style feedback."""
        attempt_text = (attempt_text or "").strip()
        step = artifacts.new_step("check_attempt")
        if not attempt_text:
            answer = "Please paste your attempt so I can check it."
            step.static_answer = answer
            progress.emit(detail="No attempt text to check — asking for it")
            result = ToolExecutionResult(
                answer=answer,
                tool_name="check_attempt",
                stream_ready=False,
                step_id=step.step_id,
            )
            return _serialize_tool_result(result)

        topic = (topic or "").strip() or chains.infer_topic_from_history(chat_history) or "General analytics"
        payload = {
            "topic": topic,
            "attempt_text": attempt_text,
            "learning_objective": chains.infer_learning_objective(topic),
            "learner_level": chains.infer_learner_level(chat_history),
            "chat_history": chains.format_chat_history(chat_history, max_messages=8),
        }
        progress.emit(
            detail=f"Reviewing your attempt on {topic} ({len(attempt_text):,} chars)"
        )
        step.practice_topic = topic
        step.stream_spec = StreamSpec(chain_key="check_chain", payload=payload)
        step.trace = {
            "tool": "check_attempt", "chain": "check_chain",
            "retrieval": False, "topic": topic,
        }

        result = ToolExecutionResult(
            answer="Prepared attempt feedback for streaming.",
            practice_topic=topic,
            tool_name="check_attempt",
            stream_ready=True,
            step_id=step.step_id,
        )
        return _serialize_tool_result(result)

    return [
        StructuredTool.from_function(
            func=answer_course_facts,
            name="answer_course_facts",
            description=(
                "Answer anything about due dates, deadlines, the class schedule, office "
                "hours, instructor or TA contact, grading weights, required materials, "
                "and which topics have been covered in class so far. Use this whenever "
                "the question is about a course fact rather than an idea."
            ),
        ),
        StructuredTool.from_function(
            func=answer_course_documents,
            name="answer_course_documents",
            description=(
                "Look up what an assignment actually requires, or what a specific class "
                "session covered. Put the TOPIC in `query`; leave `query` empty when the "
                "question is only about a period of time ('what did we cover last week') "
                "and let the date filter select the documents. "
                "Set doc_type='assignment' for assignment briefs, "
                "'announcement' for class recaps, or leave blank for both. "
                "Set days_back for recency (this week = 7, last two weeks = 14), or "
                "on_date when the student names a day -- copy their words through "
                "verbatim ('july 30'), no year needed. With on_date, set date_span "
                "to 'week' for 'the week of July 30' or 'month' for 'in July'. "
                "Do NOT use this for due dates or grading weights -- use answer_course_facts."
            ),
        ),
        StructuredTool.from_function(
            func=answer_software,
            name="answer_software",
            description=(
                "Help the student operate JMP or Excel: which menu, which dialog, "
                "which output to read. Use this for any 'how do I ... in JMP/Excel' "
                "question, including installing the software or a TreePlan add-in. "
                "Use answer_concept instead when the question is about what a "
                "statistic MEANS rather than how to produce it."
            ),
        ),
        StructuredTool.from_function(
            func=answer_concept,
            name="answer_concept",
            description=(
                "Explain what a statistic MEANS, interpretation, and analytics concepts "
                "using the Tier B concept index. Always set `module` to the ONE topic id "
                "from the module list in your instructions that best matches the question "
                "(e.g. simple-regression, inference, sensitivity-analysis). "
                "Do NOT use for assignment task lists or class recaps — use "
                "answer_course_documents. Do NOT use for JMP/Excel menus — use answer_software."
            ),
        ),
        StructuredTool.from_function(
            func=generate_practice,
            name="generate_practice",
            description="Generate one practice question for a topic. Use difficulty 'harder' for a tougher variant.",
        ),
        StructuredTool.from_function(
            func=check_attempt,
            name="check_attempt",
            description="Check a student's attempt and provide rubric-style feedback.",
        ),
    ]


def parse_tool_message_content(content: str) -> Optional[ToolExecutionResult]:
    """Parse a receipt, or None when the tool call failed.

    Returning None IS the failure signal: LangGraph turns a tool exception into
    a plain-text ToolMessage ("Error invoking tool ... Please fix the error and
    try again"), which is not our JSON. The graph's retry edge keys off exactly
    this, so a tool that raises gets the router another pass to correct itself.

    Sources and retrieval rows are no longer carried here -- they live on the
    ToolStep. See _serialize_tool_result.
    """
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "tool_name" not in payload:
        return None
    return ToolExecutionResult(
        answer=str(payload.get("answer", "")),
        abstained=bool(payload.get("abstained")),
        tool_name=str(payload.get("tool_name") or ""),
        stream_ready=bool(payload.get("stream_ready")),
        step_id=str(payload.get("step_id") or ""),
    )


def format_source_block_from_debug(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "_No sources available._"
    lines = []
    for row in rows:
        rank = row.get("rank") or "?"
        source = row.get("source") or "Unknown source"
        preview = row.get("preview") or ""
        lines.append(f"- **[{rank}] {source}**: {preview}")
    return "\n".join(lines)
