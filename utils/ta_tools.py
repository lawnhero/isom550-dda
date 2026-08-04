import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

import utils.chains_lcel as chains
from utils.retrieval import (
    build_source_block,
    hybrid_retrieve,
    retrieval_debug_rows,
    search_documents,
)


@dataclass
class StreamSpec:
    """Payload for streaming a tutoring chain in the UI after tool prep."""

    chain_key: str
    payload: Dict[str, Any]


@dataclass
class TurnArtifacts:
    """Side-channel for retrieval/diagnostics and streamable chain payloads."""

    stream_spec: Optional[StreamSpec] = None
    retrieval_debug: List[Dict[str, str]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    practice_topic: str = ""
    abstained: bool = False
    static_answer: str = ""
    tools_used: List[str] = field(default_factory=list)


@dataclass
class ToolExecutionResult:
    answer: str = ""
    sources: List[str] = field(default_factory=list)
    retrieval_debug: List[Dict[str, str]] = field(default_factory=list)
    practice_topic: str = ""
    abstained: bool = False
    tool_name: str = ""
    stream_ready: bool = False


def _serialize_tool_result(result: ToolExecutionResult) -> str:
    return json.dumps(
        {
            "answer": result.answer,
            "sources": result.sources,
            "retrieval_debug": result.retrieval_debug,
            "practice_topic": result.practice_topic,
            "abstained": result.abstained,
            "tool_name": result.tool_name,
            "stream_ready": result.stream_ready,
        }
    )


def _extract_source_labels(docs: List[Any]) -> List[str]:
    if not docs:
        return []
    block = build_source_block(docs)
    labels = []
    for line in block.splitlines():
        if line.startswith("- **["):
            start = line.find("] ") + 2
            end = line.find("**:", start)
            if end > start:
                labels.append(line[start:end])
    return labels


def _is_weak_retrieval(docs: List[Any]) -> bool:
    if not docs:
        return True
    total_chars = sum(len(getattr(doc, "page_content", "") or "") for doc in docs)
    return total_chars < 80


def _prepare_rag_tool(
    *,
    query: str,
    vector_db,
    chain_key: str,
    chat_history,
    response_mode: str,
    tool_name: str,
    artifacts: TurnArtifacts,
) -> ToolExecutionResult:
    docs = hybrid_retrieve(vector_db, query, top_k=4)
    debug_rows = retrieval_debug_rows(docs)
    artifacts.retrieval_debug = debug_rows
    artifacts.tools_used.append(tool_name)

    if _is_weak_retrieval(docs):
        answer = (
            "I don't have enough information in the course materials to answer that reliably. "
            "Please check the syllabus or ask your instructor."
        )
        artifacts.abstained = True
        artifacts.static_answer = answer
        artifacts.stream_spec = None
        return ToolExecutionResult(
            answer=answer,
            retrieval_debug=debug_rows,
            abstained=True,
            tool_name=tool_name,
            stream_ready=False,
        )

    payload = chains.build_chain_payload(
        query=query,
        chat_history=chat_history,
        response_mode=response_mode,
        context=chains._format_docs(docs),
    )
    sources = _extract_source_labels(docs)
    practice_topic = chains.infer_curriculum_topic(query)
    artifacts.sources = sources
    artifacts.practice_topic = practice_topic
    artifacts.stream_spec = StreamSpec(chain_key=chain_key, payload=payload)
    artifacts.static_answer = ""
    artifacts.abstained = False

    return ToolExecutionResult(
        answer="Prepared grounded answer for streaming.",
        sources=sources,
        retrieval_debug=debug_rows,
        practice_topic=practice_topic,
        tool_name=tool_name,
        stream_ready=True,
    )


def build_ta_tools(
    *,
    course_db,
    contents_db,
    documents_db=None,
    chains_dict: Dict[str, Any],
    chat_history,
    response_mode: str,
    artifacts: TurnArtifacts,
    course_context: str = "",
    software_context: str = "",
):
    """Build agent tools that prepare retrieval/payloads for streamed LCEL answers."""

    def answer_course_facts(query: str) -> str:
        """Answer questions about dates, people, grading, and what class has covered."""
        artifacts.tools_used.append("answer_course_facts")

        if not (course_context or "").strip():
            # No snapshot loaded. Say so rather than let another route invent a date.
            answer = (
                "I don't have the course schedule loaded right now, so I can't confirm "
                "dates or deadlines. Please check Canvas."
            )
            artifacts.static_answer = answer
            artifacts.stream_spec = None
            artifacts.abstained = True
            return _serialize_tool_result(
                ToolExecutionResult(
                    answer=answer,
                    abstained=True,
                    tool_name="answer_course_facts",
                    stream_ready=False,
                )
            )

        # No retrieval: the Tier A block already is the context.
        payload = {
            "course_context": course_context,
            "chat_history": chains.format_chat_history(chat_history, max_messages=8),
            "query": query,
        }
        artifacts.stream_spec = StreamSpec(chain_key="facts_chain", payload=payload)
        artifacts.static_answer = ""
        artifacts.abstained = False
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared course-facts answer for streaming.",
                tool_name="answer_course_facts",
                stream_ready=True,
            )
        )

    def answer_software(query: str) -> str:
        """Help operate JMP or Excel. Answers from model knowledge, no retrieval."""
        artifacts.tools_used.append("answer_software")

        # Deliberately no vector search. Retrieval here was actively harmful:
        # a JMP question used to land in answer_concept and get four unrelated
        # stats Q&A rows injected as authoritative "course context".
        artifacts.retrieval_debug = []
        artifacts.stream_spec = StreamSpec(
            chain_key="software_chain",
            payload={
                "software_context": software_context or "",
                "chat_history": chains.format_chat_history(chat_history, max_messages=8),
                "query": query,
            },
        )
        artifacts.static_answer = ""
        artifacts.abstained = False
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared software help for streaming.",
                tool_name="answer_software",
                stream_ready=True,
            )
        )

    def answer_course_documents(query: str, doc_type: str = "", days_back: int = 0) -> str:
        """Look up assignment instructions or what a class covered.

        doc_type: "assignment", "announcement", or "" for both.
        days_back: restrict to the last N days ("this week" -> 7). 0 = no limit.
        """
        artifacts.tools_used.append("answer_course_documents")

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

        docs = search_documents(
            documents_db, query, doc_type=doc_type, since_ymd=since_ymd, top_k=4
        )
        debug_rows = retrieval_debug_rows(docs)
        artifacts.retrieval_debug = debug_rows

        if _is_weak_retrieval(docs):
            answer = (
                "I couldn't find that in the class recaps or assignment instructions. "
                "Check the Canvas page for the assignment, or ask your instructor."
            )
            artifacts.abstained = True
            artifacts.static_answer = answer
            artifacts.stream_spec = None
            return _serialize_tool_result(
                ToolExecutionResult(
                    answer=answer,
                    retrieval_debug=debug_rows,
                    abstained=True,
                    tool_name="answer_course_documents",
                    stream_ready=False,
                )
            )

        sources = _extract_source_labels(docs)
        artifacts.sources = sources
        artifacts.stream_spec = StreamSpec(
            chain_key="doc_chain",
            payload={
                "context": chains._format_docs(docs),
                "chat_history": chains.format_chat_history(chat_history, max_messages=8),
                "response_mode": response_mode,
                "query": query,
            },
        )
        artifacts.static_answer = ""
        artifacts.abstained = False
        return _serialize_tool_result(
            ToolExecutionResult(
                answer="Prepared course-document answer for streaming.",
                sources=sources,
                retrieval_debug=debug_rows,
                tool_name="answer_course_documents",
                stream_ready=True,
            )
        )

    def answer_concept(query: str) -> str:
        """Explain analytics concepts, assignment help, coding, or interpretation using class materials."""
        result = _prepare_rag_tool(
            query=query,
            vector_db=contents_db,
            chain_key="step_chain",
            chat_history=chat_history,
            response_mode=response_mode,
            tool_name="answer_concept",
            artifacts=artifacts,
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
        artifacts.tools_used.append("generate_practice")
        artifacts.practice_topic = topic
        artifacts.stream_spec = StreamSpec(chain_key="practice_chain", payload=payload)
        artifacts.static_answer = ""
        artifacts.abstained = False

        result = ToolExecutionResult(
            answer="Prepared practice question for streaming.",
            practice_topic=topic,
            tool_name="generate_practice",
            stream_ready=True,
        )
        return _serialize_tool_result(result)

    def check_attempt(attempt_text: str, topic: str = "") -> str:
        """Check a student's attempt and return rubric-style feedback."""
        attempt_text = (attempt_text or "").strip()
        artifacts.tools_used.append("check_attempt")
        if not attempt_text:
            answer = "Please paste your attempt so I can check it."
            artifacts.static_answer = answer
            artifacts.stream_spec = None
            result = ToolExecutionResult(
                answer=answer,
                tool_name="check_attempt",
                stream_ready=False,
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
        artifacts.practice_topic = topic
        artifacts.stream_spec = StreamSpec(chain_key="check_chain", payload=payload)
        artifacts.static_answer = ""
        artifacts.abstained = False

        result = ToolExecutionResult(
            answer="Prepared attempt feedback for streaming.",
            practice_topic=topic,
            tool_name="check_attempt",
            stream_ready=True,
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
                "session covered. Set doc_type='assignment' for assignment briefs, "
                "'announcement' for class recaps, or leave blank for both. Set "
                "days_back to restrict by recency (this week = 7, last two weeks = 14). "
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
                "Explain analytics concepts, assignment help, coding, and interpretation using class materials."
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
    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or "tool_name" not in payload:
        return None
    raw_debug = payload.get("retrieval_debug") or []
    retrieval_debug = []
    if isinstance(raw_debug, list):
        for row in raw_debug:
            if isinstance(row, dict):
                retrieval_debug.append(
                    {
                        "rank": str(row.get("rank", "")),
                        "source": str(row.get("source", "")),
                        "preview": str(row.get("preview", "")),
                    }
                )
    return ToolExecutionResult(
        answer=str(payload.get("answer", "")),
        sources=list(payload.get("sources") or []),
        retrieval_debug=retrieval_debug,
        practice_topic=str(payload.get("practice_topic") or ""),
        abstained=bool(payload.get("abstained")),
        tool_name=str(payload.get("tool_name") or ""),
        stream_ready=bool(payload.get("stream_ready")),
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
