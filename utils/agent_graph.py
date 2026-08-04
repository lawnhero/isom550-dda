from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from utils.ta_tools import (
    ToolExecutionResult,
    TurnArtifacts,
    build_ta_tools,
    parse_tool_message_content,
)


AGENT_SYSTEM_PROMPT = """You are Dayton, the Virtual TA for ISOM 550 Data and Decision Analytics.

Choose the best tool for each student request:
- answer_course_facts: due dates, deadlines, schedule, office hours, instructor or TA
  contact, grading weights, required materials, what has been covered in class so far
- answer_course_documents: what an assignment requires, or what a specific class
  session covered. Set doc_type='assignment' or 'announcement' to narrow it, and
  days_back to restrict by recency (this week = 7, last two weeks = 14)
- answer_software: how to DO something in JMP or Excel -- menus, dialogs, reading
  output, installing the software or the TreePlan add-in
- answer_concept: what a statistic MEANS, interpretation, analytics concepts
- generate_practice: when the student wants a practice question
- check_attempt: when the student wants feedback on their attempt

Rules:
1) Any question about a date, a person, or grading must use answer_course_facts.
2) "What did we cover" questions: use answer_course_facts for the list of class
   topics, and answer_course_documents with doc_type='announcement' when the
   student wants detail about what was actually taught in a session.
3) "What am I supposed to do for <assignment>" is answer_course_documents with
   doc_type='assignment', not answer_concept.
4) "How do I ... in JMP/Excel" is answer_software. "What does this coefficient
   mean" is answer_concept. A question can need both -- if so, call both.
5) Prefer one primary tool per turn unless a short follow-up tool call clearly helps.
6) After a tool returns, reply with a very short acknowledgement only (one short sentence).
   The student-facing tutoring answer is streamed separately from the tool payload.
7) Do not invent course policies, deadlines, or grading rules.
8) Keep responses concise and student-friendly.
"""


def _history_to_messages(chat_history, max_messages: int = 8) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    if not chat_history:
        return messages
    for message in chat_history[-max_messages:]:
        if "Human" in str(type(message)):
            messages.append(HumanMessage(content=message.content))
        else:
            messages.append(AIMessage(content=message.content))
    return messages


def build_ta_agent(
    *,
    agent_llm: BaseLanguageModel,
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
    tools = build_ta_tools(
        course_db=course_db,
        contents_db=contents_db,
        documents_db=documents_db,
        chains_dict=chains_dict,
        chat_history=chat_history,
        response_mode=response_mode,
        artifacts=artifacts,
        course_context=course_context,
        software_context=software_context,
    )
    return create_react_agent(agent_llm, tools=tools, prompt=AGENT_SYSTEM_PROMPT)


def _collect_tool_results(messages: List[BaseMessage]) -> List[ToolExecutionResult]:
    results: List[ToolExecutionResult] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        parsed = parse_tool_message_content(message.content)
        if parsed:
            results.append(parsed)
    return results


def _extract_tool_calls(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """Return ordered tool calls (name + args) from agent AIMessages in this turn."""
    tool_calls: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            if not name:
                continue
            tool_calls.append(
                {
                    "name": name,
                    "args": args or {},
                }
            )
    return tool_calls


def _final_answer(messages: List[BaseMessage], tool_results: List[ToolExecutionResult]) -> str:
    for item in reversed(tool_results):
        if item.answer and not item.stream_ready:
            return item.answer
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            content = message.content
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                joined = "\n".join(part for part in text_parts if part).strip()
                if joined:
                    return joined
    return "I could not generate a response for that request."


def run_ta_turn(
    *,
    agent,
    query: str,
    chat_history,
    artifacts: TurnArtifacts,
    memory_window: int = 8,
    recursion_limit: int = 4,
) -> Dict[str, Any]:
    prior_messages = _history_to_messages(chat_history, max_messages=memory_window)
    result = agent.invoke(
        {"messages": prior_messages + [HumanMessage(content=query)]},
        config={"recursion_limit": recursion_limit},
    )
    messages = result.get("messages", [])
    tool_results = _collect_tool_results(messages)
    tool_calls = _extract_tool_calls(messages)
    tools_used = [item["name"] for item in tool_calls] or list(artifacts.tools_used)

    sources = artifacts.sources
    retrieval_debug = artifacts.retrieval_debug
    practice_topic = artifacts.practice_topic
    abstained = artifacts.abstained
    for item in tool_results:
        if item.sources and not sources:
            sources = item.sources
        if item.retrieval_debug and not retrieval_debug:
            retrieval_debug = item.retrieval_debug
        if item.practice_topic and not practice_topic:
            practice_topic = item.practice_topic
        if item.abstained:
            abstained = True

    stream_spec = artifacts.stream_spec
    if stream_spec is not None:
        answer = ""
    elif artifacts.static_answer:
        answer = artifacts.static_answer
    else:
        answer = _final_answer(messages, tool_results)

    route_label = tools_used[0] if tools_used else "agent_direct"

    return {
        "answer": answer,
        "stream_spec": stream_spec,
        "tools_used": tools_used,
        "tool_calls": tool_calls,
        "route_label": route_label,
        "sources": sources,
        "retrieval_debug": retrieval_debug,
        "practice_topic": practice_topic,
        "abstained": abstained,
        "messages": messages,
    }
