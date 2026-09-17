import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

import utils.chains_lcel as chains
import utils.ui as ui
from utils import practice
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
    """Payload for streaming a tutoring chain in the UI after tool prep.

    `images` is the side channel for screenshots. They are already inside
    `payload` for the chain to render, and are repeated here so the turn can be
    inspected (and logged, by count) without unpacking a chain payload.
    """

    chain_key: str
    payload: Dict[str, Any]
    images: List[Dict[str, str]] = field(default_factory=list)


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
    # One line saying what this section answers, in the student's terms. Read
    # by annotate_compound_turn so the OTHER sections of a two-tool turn can
    # be told what not to cover. Usually the router's own query argument.
    covers: str = ""
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

    The router is told a question can need two tools ("how do I run a
    regression in JMP, and what does R-squared mean?"), so each call gets its
    own ToolStep rather than sharing one set of slots. Keyed by a per-call
    `step_id` the tool generates and echoes back in its receipt, so the caller
    can match each receipt to the payload it belongs to (see router.run_ta_turn).
    """

    steps: Dict[str, ToolStep] = field(default_factory=dict)

    def new_step(self, tool_name: str) -> ToolStep:
        """Claim a fresh slot for one tool call."""
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
    # Echoed back through the receipt so run_ta_turn can match this result
    # to the ToolStep that holds its (unserialisable) chain payload.
    step_id: str = ""


def _serialize_tool_result(result: ToolExecutionResult) -> str:
    """What a tool returns. Deliberately a receipt, not the answer.

    The router never reads retrieved text or the chain payload -- the tutoring
    answer is streamed from the ToolStep by app.py. `step_id` correlates this
    receipt to the ToolStep holding that payload; the rest says whether the
    call succeeded. Everything else is read off the ToolStep instead.
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
        f"Summarise what these {kind}{when} covered. List every one by name and "
        "say briefly what it was about."
    )


def _squash(text: str) -> str:
    return " ".join((text or "").split()).lower()


def _attachment_bodies(attachment_text: str) -> List[str]:
    """The squashed body of each attached file, marker lines removed."""
    bodies = []
    for block in (attachment_text or "").split("--- Attached file:"):
        body = block.split("---", 1)[-1] if "---" in block else block
        body = _squash(body)
        if body:
            bodies.append(body)
    return bodies


def _merge_attachment(attempt_text: str, attachment_text: str) -> str:
    """The attempt the checker should grade, given what the router wrote and
    what the student actually attached.

    The router is told not to copy attachments, and does anyway -- observed:
    882 of 1,925 characters, then it stopped. So the attachment in the closure
    is authoritative, and the router's text is kept only where it adds the
    student's own words:

      attachment fully inside attempt   -> already there, keep the attempt
      attempt inside the attachment     -> a (partial) copy, replace it
      otherwise                         -> their own words plus a copy, or
                                           no copy; append the attachment.
                                           Some duplication beats truncation.

    Whitespace-normalised throughout, because a copying model drops the
    marker line and reflows the text.
    """
    attempt_text = (attempt_text or "").strip()
    if not attachment_text:
        return attempt_text
    attempt = _squash(attempt_text)
    bodies = _attachment_bodies(attachment_text)
    if attempt and all(body in attempt for body in bodies):
        return attempt_text
    if attempt and any(attempt in body for body in bodies):
        return attachment_text
    return f"{attempt_text}\n\n{attachment_text}" if attempt_text else attachment_text


def _concept_focus(found: RetrievalResult) -> str:
    """The concept a strong Tier B hit was about, as a practice topic.

    "Practice this" used to drill a keyword-inferred curriculum label:
    "What does an R-squared of 0.62 mean?" matched the word "r-squared" under
    "Regression", and the chip composed a request to practise Regression --
    while the tool had just retrieved the concept titled "Interpreting
    R-squared" at distance 0.43. The retrieved concept IS the topic; the six
    hand-written labels are the wrong granularity (and the module is only
    slightly better -- multiple-regression holds seven concepts).
    """
    if not found.docs:
        return ""
    metadata = getattr(found.docs[0], "metadata", {}) or {}
    if not metadata.get("concept_id"):
        return ""
    if found.quality == "strong":
        return str(metadata.get("title") or "").strip()
    # A loose hit still landed in a module; that is a better topic than a
    # keyword guess at the question, and it comes from the same retrieval.
    from utils.concept_taxonomy import module_label

    return module_label(str(metadata.get("module") or ""))


# Grounding a practice question in a concept needs a much closer match than
# answering one. A student's question is a sentence; a practice topic is two
# or three words, and short strings sit closer to everything: "Regression"
# alone scores 1.24 against an unrelated concept, "Bayes theorem" 1.21 --
# both inside the 1.40 "strong" band. A concept title scores ~0.4-0.6.
PRACTICE_GROUND_MAX_DISTANCE = 1.0
# When the topic came from a pill it arrives as "Module: Topic", and the
# module becomes a filter. Inside the right module the nearest concept is the
# right one by construction, so the bar can be looser: measured across every
# pill, "Simple regression: Slope" sits at 0.73 with the filter and the worst
# ("Multiple regression: Model building ...") at 1.07.
PRACTICE_GROUND_MAX_DISTANCE_IN_MODULE = 1.2


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

# What check_chain sees in place of a practice question when none is open.
# Most attempts are a student's own assignment work, not an answer to a
# generated drill, so an empty slot is the normal case and must not read as a
# missing one -- the prompt used to tell the model to announce it was grading
# without the question, which is noise on every homework check.
NO_HELD_QUESTION = (
    "(No practice question is open. The student is checking their own work: "
    "take the task from their attempt and the recent chat.)"
)


def _stream_spec(
    chain_key: str,
    payload: Dict[str, Any],
    images: Optional[List[Dict[str, str]]] = None,
) -> StreamSpec:
    """Point a step at the plain chain, or at its vision build when there are
    screenshots to show it.

    The `_vision` suffix is the whole routing rule -- see
    `chains_lcel.get_all_chains`. A chain without one simply never receives an
    image, which is why this is safe to call from any tool.
    """
    if not images:
        return StreamSpec(chain_key=chain_key, payload=payload)
    return StreamSpec(
        chain_key=f"{chain_key}_vision",
        payload={**payload, "images": images},
        images=list(images),
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
    module: str = "",
    memory_window: int = chains.DEFAULT_MEMORY_WINDOW,
    images: Optional[List[Dict[str, str]]] = None,
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

    if found.quality == "none":
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
        memory_window=memory_window,
    )
    sources = _extract_source_labels(found.docs)
    practice_topic = _concept_focus(found) or chains.infer_curriculum_topic(query)
    step.sources = sources
    step.practice_topic = practice_topic
    step.covers = query
    step.stream_spec = _stream_spec(chain_key, payload, images)
    step.retrieval_quality = found.quality
    step.trace = {
        "tool": tool_name, "chain": step.stream_spec.chain_key, "retrieval": True,
        "hits": len(found.docs), "topic": practice_topic or "(none)",
        "images": len(images or []),
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


# Word budget per section of a compound turn. Two sections at the single-turn
# caps (180-200 words each, and software_chain had none) made a 500-word
# reply; two parts of one answer should read like one answer.
COMPOUND_SECTION_WORDS = 150


def annotate_compound_turn(steps: List[ToolStep]) -> int:
    """Tell each streamed section that it is one part of a larger answer.

    Writes `turn_context` into the payload of every step that will stream,
    when there are at least two of them; every chain template carries a
    `{turn_context}` slot that renders this block (see chains_lcel). Returns
    the number of parts annotated, 0 when the turn is not compound.

    The chains were written to answer alone, and alone is what they did even
    when paired: the JMP section explained R-squared before the concept
    section got to, both signed off with their own follow-up question, and
    nothing told the student where one part ended and the next began. The
    router already knows the split -- it wrote a query per tool -- so each
    section is told what the others cover and what that leaves for it.

    Static answers (abstentions, "paste your attempt") are not parts; a turn
    with one streamed section and one refusal is not compound.
    """
    streamed = [step for step in steps if step.stream_spec is not None]
    total = len(streamed)
    if total < 2:
        return 0
    for index, step in enumerate(streamed, start=1):
        lines = [
            f"THIS IS PART {index} OF {total} OF ONE ANSWER. The student asked a "
            "compound question; each part is written separately and read in order.",
        ]
        for other_index, other in enumerate(streamed, start=1):
            who = "you" if other is step else "written separately"
            what = other.covers or ui.route_meta(other.tool_name)["badge"]
            lines.append(f"- Part {other_index} ({who}): {what}")
        lines.append("Rules for your part. These override any length or ending rule above:")
        lines.append("- Open with a bold heading of at most six words naming what this part covers.")
        lines.append(
            "- Cover only your part. Do not explain what another part covers; "
            "refer to it in a few words at most (\"see the R-squared part below\")."
        )
        lines.append("- Do not greet, introduce yourself, or restate the question.")
        lines.append(f"- Keep your part under {COMPOUND_SECTION_WORDS} words.")
        if index == total:
            lines.append("- You are the last part: end with one short follow-up question.")
        else:
            lines.append(
                "- You are not the last part: end with your content. No closing "
                "question, offer, or summary -- the next part follows immediately."
            )
        step.stream_spec.payload["turn_context"] = "\n".join(lines)
    return total


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
    memory_window: int = chains.DEFAULT_MEMORY_WINDOW,
    images: Optional[List[Dict[str, str]]] = None,
    practice_session: Optional[Dict[str, Any]] = None,
    attachment_text: str = "",
):
    """Build agent tools that prepare retrieval/payloads for streamed LCEL answers.

    `practice_session` is the question currently on the student's screen (see
    utils/practice.py). The tools READ it; app.py writes it after the stream,
    because the question text does not exist until the stream has produced it.

    `attachment_text` is the decoded content of any text/PDF the student
    attached (see attachments.extract_attachments). It reaches check_attempt
    through this closure, the same way screenshots do, rather than by asking
    the router to copy it into `attempt_text`: the router writes tool calls
    under an output-token cap, and an attached file can be thousands of
    characters, so copying truncated the call into invalid JSON.

    """
    # Screenshots reach the chains through the closure, not through tool
    # arguments. The router never sees the pixels -- it gets a one-line marker
    # in the query (see attachments.describe_images) and picks a route from
    # that -- so the agent loop is not paying image tokens on every hop, and a
    # model cannot mangle a base64 blob while writing a tool call.
    images = list(images or [])
    attachment_text = (attachment_text or "").strip()

    def _history_text() -> str:
        return chains.format_chat_history(chat_history, max_messages=memory_window)

    def answer_course_facts(query: str) -> str:
        """Answer questions about dates, people, grading, and what class has covered.

        Args:
            query: The student's question. The Tier A snapshot is read whole, so
                this is the question itself rather than search terms.
        """
        step = artifacts.new_step("answer_course_facts")

        if not (course_context or "").strip():
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

        # No retrieval: the Tier A block already is the context.
        payload = {
            "course_context": course_context,
            "chat_history": _history_text(),
            "query": query,
        }
        step.covers = query
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
        """Help operate JMP or Excel. Answers from model knowledge, no retrieval.

        Args:
            query: What the student is trying to do in the tool, including the
                task and any output they are looking at.
        """
        step = artifacts.new_step("answer_software")

        # Deliberately no vector search. Retrieval here was actively harmful:
        # a JMP question used to land in answer_concept and get four unrelated
        # stats Q&A rows injected as authoritative "course context".
        step.covers = query
        step.stream_spec = _stream_spec(
            "software_chain",
            {
                "software_context": software_context or "",
                "chat_history": _history_text(),
                "query": query,
            },
        )
        step.trace = {
            "tool": "answer_software", "chain": step.stream_spec.chain_key,
            "retrieval": False, "context_chars": len(software_context or ""),
            "images": len(images),
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

        Args:
            query: The topic to search for. MAY BE EMPTY when the question is
                purely about a date range ("what did we cover last week") -- the
                filter alone then selects the documents.
            doc_type: "assignment" for assignment briefs, "announcement" for
                class recaps, or "" for both.
            days_back: Restrict to the last N days ("this week" -> 7, "last two
                weeks" -> 14). 0 = no limit.
            on_date: A date the student named, copied verbatim from their
                question ("july 30", "7/30"). No year needed.
            date_span: Window around on_date -- "day" for a single named day
                ("on July 21"); "week" for a window ("around July 21", "the week
                of July 30", "that week"); "month" for a whole month ("in July").
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
        doc_question = (query or "").strip() or _filter_only_question(
            doc_type, days_back, on_date, date_span
        )
        step.covers = doc_question
        step.stream_spec = StreamSpec(
            chain_key="doc_chain",
            payload={
                "context": chains._format_docs(found.docs),
                "chat_history": _history_text(),
                "response_mode": response_mode,
                "query": doc_question,
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
        """Explain analytics concepts, interpretation, and what a statistic means.

        Args:
            query: The concept question the student is asking.
            module: ONE module id from the Tier B module list in your
                instructions (e.g. simple-regression, inference,
                sensitivity-analysis). Narrows the search to that module.
        """
        result = _prepare_rag_tool(
            query=query,
            vector_db=contents_db,
            chain_key="concept_chain",
            chat_history=chat_history,
            response_mode=response_mode,
            tool_name="answer_concept",
            artifacts=artifacts,
            module=module,
            memory_window=memory_window,
            images=images,
        )
        return _serialize_tool_result(result)

    def generate_practice(topic: str, difficulty: str = "same") -> str:
        """Generate one MBA-style practice question for a topic.

        Args:
            topic: What to drill, as specifically as the student named it --
                "Interpreting R-squared", not "Regression"; "folding back a
                decision tree", not "Decision analysis". Leave empty to reuse
                the topic already under discussion in the conversation.
            difficulty: "same" (default), or "harder" for a tougher variant of
                the topic the student just practised.
        """
        topic = (topic or "").strip() or chains.infer_topic_from_history(chat_history) or "General analytics"
        difficulty = (difficulty or "same").strip().lower()
        if difficulty not in practice.DIFFICULTIES:
            difficulty = "same"
        step = artifacts.new_step("generate_practice")

        # Ground the question in the concept the topic names, when the index
        # has one that close. The concept row carries the instructor's own
        # framing and the common student mistake, which is what separates a
        # drill on THIS course's R-squared from a generic one. A topic that
        # matches nothing closely (a bare "Regression") gets no grounding
        # rather than a loosely related concept's mistake to target.
        concept_context = ""
        grounded_on = ""
        module_id = ""
        if contents_db is not None:
            from utils.concept_taxonomy import split_focus

            # "Simple regression: R-squared and model fit" names its module,
            # and the module is a FILTER here, not the topic. Without it that
            # exact string grounded on a logistic-regression concept (0.89,
            # inside the bar) because "model" and "regression" pulled harder
            # than "R-squared".
            module_id, _ = split_focus(topic)
            found = search_concepts(
                contents_db, topic, module=module_id, top_k=1, in_conversation=True
            )
            bar = (
                PRACTICE_GROUND_MAX_DISTANCE_IN_MODULE if module_id
                else PRACTICE_GROUND_MAX_DISTANCE
            )
            if found.docs and found.best_distance <= bar:
                concept_context = chains._format_docs(found.docs)
                grounded_on = _extract_source_labels(found.docs)[0]
                step.sources = [grounded_on]
                step.retrieval_debug = retrieval_debug_rows(found.docs)

        payload = {
            "topic": topic,
            "difficulty": difficulty,
            "chat_history": _history_text(),
            # So a "harder" variant escalates the question on screen instead of
            # silently restating it.
            "previous_question": practice.question_of(practice_session),
            "concept_context": concept_context or "(none — write from your own knowledge of the topic)",
        }
        # A student who has asked for help on the live question and then gets a
        # brand-new one has almost certainly been misrouted: they said "I'm
        # stuck", not "give me another". Not blocked here -- the router is an
        # LLM and a hard block would break a genuine "ask me a different one" --
        # but recorded, so the rate is measurable before anything is hardened.
        probable_misroute = (
            practice.is_active(practice_session)
            and int(practice_session.get("hints_given") or 0) > 0
        )
        step.practice_topic = topic
        step.covers = f"a new practice question on {topic}"
        step.stream_spec = StreamSpec(chain_key="practice_chain", payload=payload)
        step.trace = {
            "tool": "generate_practice", "chain": "practice_chain",
            "retrieval": bool(concept_context), "topic": topic,
            "module": module_id or "(none)",
            "difficulty": difficulty, "grounded_on": grounded_on or "(none)",
            "probable_misroute": probable_misroute,
        }

        result = ToolExecutionResult(
            answer="Prepared practice question for streaming.",
            practice_topic=topic,
            tool_name="generate_practice",
            stream_ready=True,
            step_id=step.step_id,
        )
        return _serialize_tool_result(result)

    def coach_practice(query: str, request: str = "hint") -> str:
        """Help the student with the practice question already on their screen.

        Args:
            query: The student's message, copied verbatim.
            request: "hint" for one nudge toward the next move, "clarify" to
                explain what the question is asking without moving toward the
                answer, or "worked_step" to work one step and stop short of the
                result. Defaults to "hint".
        """
        step = artifacts.new_step("coach_practice")
        if not practice.is_active(practice_session):
            # Coaching nothing is worse than admitting there is nothing to
            # coach: the student would get a confident hint about a question
            # that is not on their screen.
            answer = (
                "I don't have a practice question open right now. Ask me for "
                "one and I'll walk you through it."
            )
            step.static_answer = answer
            step.trace = {
                "tool": "coach_practice", "chain": None, "retrieval": False,
                "practice_active": False,
            }
            return _serialize_tool_result(
                ToolExecutionResult(
                    answer=answer,
                    tool_name="coach_practice",
                    stream_ready=False,
                    step_id=step.step_id,
                )
            )

        # What the student asked for is not always what they are owed: a fourth
        # hint on a question they have been stuck on three times is stalling.
        resolved = practice.effective_request(practice_session, request)
        topic = practice.topic_of(practice_session) or "General analytics"
        payload = {
            "topic": topic,
            "request": resolved,
            "question_block": practice.prompt_block(practice_session),
            "chat_history": _history_text(),
            "query": query or "",
        }
        step.practice_topic = topic
        step.covers = f"help ({resolved.replace('_', ' ')}) with the practice question on screen"
        step.stream_spec = StreamSpec(chain_key="coach_chain", payload=payload)
        step.trace = {
            "tool": "coach_practice", "chain": "coach_chain", "retrieval": False,
            "topic": topic, "requested": request, "resolved": resolved,
            "hints_given": practice_session.get("hints_given", 0),
            "practice_active": True,
        }
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared coaching for streaming.",
                practice_topic=topic,
                tool_name="coach_practice",
                stream_ready=True,
                step_id=step.step_id,
            )
        )

    def check_attempt(attempt_text: str, topic: str = "") -> str:
        """Check a student's attempt and return rubric-style feedback.

        Args:
            attempt_text: What the student TYPED, copied verbatim. Do not copy
                the contents of any "--- Attached file: ... ---" block -- the
                attached file is handed to the checker automatically, so pass
                only the student's own words (or leave this empty when the
                attempt is entirely in the attachment). If the work is in a
                screenshot, the image is shown to the chain either way; do
                not describe or transcribe it here.
            topic: What the attempt is about. Leave empty to infer it from the
                conversation.
        """
        attempt_text = (attempt_text or "").strip()
        step = artifacts.new_step("check_attempt")
        # The attached file is the attempt, or most of it, and it comes from
        # the closure -- never trust the router's copy of it.
        attempt_text = _merge_attachment(attempt_text, attachment_text)
        if not attempt_text and images:
            # An uploaded screenshot with no typed words is the single most
            # common way a student says "check my work". Asking them to paste
            # an attempt we can already see would be absurd.
            attempt_text = (
                "(The student's work is in the attached screenshot, "
                "not in text.)"
            )
        if not attempt_text:
            answer = "Please paste your attempt so I can check it."
            step.static_answer = answer
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
            "chat_history": _history_text(),
            # The question itself, not a recollection of it. Held in session
            # state because the recent-chat window is 8 messages -- four turns --
            # and question/hint/clarify/attempt is exactly four.
            "question": practice.question_of(practice_session) or NO_HELD_QUESTION,
        }
        step.practice_topic = topic
        step.covers = f"feedback on the student's attempt ({topic})"
        step.stream_spec = _stream_spec("check_chain", payload, images)
        step.trace = {
            "tool": "check_attempt", "chain": step.stream_spec.chain_key,
            "retrieval": False, "topic": topic,
            "images": len(images),
            "held_question": bool(practice.question_of(practice_session)),
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
            parse_docstring=True,
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
            parse_docstring=True,
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
            parse_docstring=True,
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
            parse_docstring=True,
        ),
        StructuredTool.from_function(
            func=generate_practice,
            name="generate_practice",
            description="Generate one practice question for a topic. Use difficulty 'harder' for a tougher variant.",
            parse_docstring=True,
        ),
        StructuredTool.from_function(
            func=coach_practice,
            name="coach_practice",
            description=(
                "Help with the practice question ALREADY on the student's screen. "
                "Use this for every 'I'm stuck', 'give me a hint', 'I don't "
                "understand', 'this is confusing' or 'what is it asking' while a "
                "practice question is open -- never generate_practice, which would "
                "replace the question they are working on. Set request='hint' for a "
                "nudge, 'clarify' to explain what the question is asking, or "
                "'worked_step' to work one step for them."
            ),
            parse_docstring=True,
        ),
        StructuredTool.from_function(
            func=check_attempt,
            name="check_attempt",
            description="Check a student's attempt and provide rubric-style feedback.",
            parse_docstring=True,
        ),
    ]


def parse_tool_message_content(content: str) -> Optional[ToolExecutionResult]:
    """Parse a receipt, or None when the tool call failed.

    Returning None IS the failure signal: anything that is not our JSON receipt
    means the tool did not complete, and router.run_ta_turn drops that call.

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


