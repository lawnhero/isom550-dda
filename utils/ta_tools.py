import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool

import utils.chains_lcel as chains
from utils.retrieval import build_source_block, hybrid_retrieve, retrieval_debug_rows


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
    memory_summary: str,
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
        memory_summary=memory_summary,
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
    chains_dict: Dict[str, Any],
    chat_history,
    memory_summary: str,
    response_mode: str,
    artifacts: TurnArtifacts,
):
    """Build agent tools that prepare retrieval/payloads for streamed LCEL answers."""

    def answer_logistics(query: str) -> str:
        """Answer course logistics questions using syllabus, grading, and schedule materials."""
        result = _prepare_rag_tool(
            query=query,
            vector_db=course_db,
            chain_key="rag_chain",
            chat_history=chat_history,
            memory_summary=memory_summary,
            response_mode="Direct answer",
            tool_name="answer_logistics",
            artifacts=artifacts,
        )
        return _serialize_tool_result(result)

    def answer_concept(query: str) -> str:
        """Explain analytics concepts, assignment help, coding, or interpretation using class materials."""
        result = _prepare_rag_tool(
            query=query,
            vector_db=contents_db,
            chain_key="step_chain",
            chat_history=chat_history,
            memory_summary=memory_summary,
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
            "memory_summary": memory_summary or "No memory summary yet.",
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
            "memory_summary": memory_summary or "No memory summary yet.",
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
            func=answer_logistics,
            name="answer_logistics",
            description=(
                "Answer deadlines, grading, schedule, policy, and syllabus logistics using course materials."
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
