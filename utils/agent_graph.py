import time
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from utils.progress import (
    PHASE_RETRY,
    PHASE_ROUTING,
    ProgressReporter,
)
from utils.ta_tools import (
    ToolExecutionResult,
    ToolStep,
    TurnArtifacts,
    build_ta_tools,
    parse_tool_message_content,
)
from utils.chains_lcel import DEFAULT_MEMORY_WINDOW


AGENT_SYSTEM_PROMPT = """You are Dayton, the Virtual TA for ISOM 550 Data and Decision Analytics.

Choose the best tool for each student request:
- answer_course_facts: due dates, deadlines, schedule, office hours, instructor or TA
  contact, grading weights, required materials, what has been covered in class so far
- answer_course_documents: what an assignment requires, or what a specific class
  session covered. Set doc_type='assignment' or 'announcement' to narrow it, and
  days_back to restrict by recency (this week = 7, last two weeks = 14)
- answer_software: how to DO something in JMP or Excel -- menus, dialogs, reading
  output, installing the software or the TreePlan add-in
- answer_concept: what a statistic MEANS, interpretation, analytics concepts.
  Always pass `module` — one topic id from the Tier B module list below
- generate_practice: when the student wants a practice question
- check_attempt: when the student wants feedback on their attempt

Rules:
1) Course facts vs documents — pick one primary route:
   - answer_course_facts: deadlines, due dates, schedule, office hours, people,
     grading weights, required materials, and the overview of which class topics
     have been covered so far.
   - answer_course_documents: what an assignment requires, or what was taught in
     a specific class session. NOT for due dates or grading weights.
   - Deadline gate: if the student asks WHEN something is due ("due", "deadline",
     "when is ... due"), use answer_course_facts and pass the whole question as
     query. This holds even when they also name an assignment or a date --
     "what assignment is due around July 22" is answer_course_facts, not
     answer_course_documents. Only the synced schedule may state a due date.
2) "What did we cover" — two different questions:
   - Overview ("what topics have we done", "what have we covered so far") →
     answer_course_facts.
   - Session detail ("what did we learn on July 30", "what happened in class
     last week") → answer_course_documents with doc_type='announcement'.
     Copy a named date verbatim into on_date ('july 30', '7/30'). Set date_span:
     - 'day' only when they mean that single day ("on July 21", "July 21 class").
     - 'week' when they mean a window ("around July 21", "the week of July 30",
       "that week"). This is the ISO week containing the named date (Mon–Sun).
     - 'month' when they mean a whole month ("in July").
     Leave query empty when the question is only about a time period and use
     days_back or on_date to select documents.
3) "What am I supposed to do for <assignment>" → answer_course_documents with
   doc_type='assignment', not answer_concept.
4) Software vs concepts:
   - "How do I ... in JMP/Excel" → answer_software.
   - "What does this coefficient/statistic/output mean" → answer_concept.
   - When both parts are asked ("how do I run it, and what does R-squared mean"),
     call both tools in one turn.
5) Practice and attempts:
   - generate_practice when the student wants a drill question.
   - check_attempt when they want feedback on work they wrote. Pass the full
     attempt in attempt_text, including any attached file content in the message.
6) Attachments:
   - Blocks marked "--- Attached file: ... ---" are the student's own work or
     data, not course material. Route to check_attempt when they want it reviewed;
     copy the attached content into attempt_text.
7) You are a dispatcher, not the writer. The student-facing answer is streamed
   from the tool; anything you write yourself is shown ONLY when you call no
   tool. Do not summarise, preview, or restate what a tool will say.
8) When you call no tool (greetings, thanks, or meta questions about what you
   can do): reply briefly in character. Do not invent course policies, deadlines,
   or grading rules — suggest a concrete question instead.
9) Keep your own replies concise and student-friendly.
"""


def _history_to_messages(chat_history, max_messages: int = DEFAULT_MEMORY_WINDOW) -> List[BaseMessage]:
    messages: List[BaseMessage] = []
    if not chat_history:
        return messages
    for message in chat_history[-max_messages:]:
        if "Human" in str(type(message)):
            messages.append(HumanMessage(content=message.content))
        else:
            messages.append(AIMessage(content=message.content))
    return messages


def _agent_system_prompt() -> str:
    from utils.concept_taxonomy import format_modules_for_prompt

    return (
        AGENT_SYSTEM_PROMPT
        + "\n\nTier B modules — pass ONE `module` id to answer_concept:\n"
        + format_modules_for_prompt()
    )


def build_ta_agent(
    *,
    agent_llm: BaseLanguageModel,
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
    system_prompt: Optional[str] = None,
    memory_window: int = DEFAULT_MEMORY_WINDOW,
    images: Optional[List[Dict[str, str]]] = None,
):
    tools = build_ta_tools(
        contents_db=contents_db,
        documents_db=documents_db,
        chains_dict=chains_dict,
        chat_history=chat_history,
        response_mode=response_mode,
        artifacts=artifacts,
        course_context=course_context,
        software_context=software_context,
        course_span=course_span,
        progress=progress,
        memory_window=memory_window,
        # Handed to the tools, never to the router: `agent_llm` only has to
        # choose a route, and the query already carries a one-line marker
        # saying a screenshot is present and readable.
        images=images,
    )
    return _build_graph(
        agent_llm,
        tools,
        system_prompt=system_prompt or _agent_system_prompt(),
    )


def _tool_call_failed(message: ToolMessage) -> bool:
    """True when a tool raised instead of returning a receipt.

    Two independent signals, because neither is guaranteed alone: LangGraph
    marks converted exceptions with status="error", and our own receipts are
    always JSON that parse_tool_message_content accepts. Anything that is
    neither is a failure.
    """
    if getattr(message, "status", None) == "error":
        return True
    return parse_tool_message_content(message.content) is None


def _after_tools(state: MessagesState) -> str:
    """Go back to the router only if a tool call needs correcting.

    This edge is the reason the graph is hand-built. The prebuilt ReAct agent
    wires `tools -> agent` unconditionally, so EVERY turn paid for a second
    router call whose output was then discarded -- roughly a second of latency
    and a full LLM round-trip per question, for nothing.

    The one case that genuinely needs the extra pass is a tool call that failed
    validation, where LangGraph hands the model "Please fix the error and try
    again" and it can retry with corrected arguments. That is what the retry
    budget in `recursion_limit` exists for.
    """
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break  # walked past this batch of tool results
        if _tool_call_failed(message):
            return "agent"
    return END


def _build_graph(agent_llm: BaseLanguageModel, tools, system_prompt: str = AGENT_SYSTEM_PROMPT):
    """agent -> tools -> (retry only on failure) -> END.

    Deliberately explicit rather than a prebuilt ReAct agent, so the control
    flow is visible and the retry condition is ours to set.
    """
    model = agent_llm.bind_tools(tools)

    def agent(state: MessagesState) -> Dict[str, Any]:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages
        return {"messages": [model.invoke(messages)]}

    def route_from_agent(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_from_agent, {"tools": "tools", END: END})
    graph.add_conditional_edges("tools", _after_tools, {"agent": "agent", END: END})
    return graph.compile()


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


def _ordered_steps(
    tool_results: List[ToolExecutionResult], artifacts: TurnArtifacts
) -> List[ToolStep]:
    """Put the prepared work back into the order the model asked for it.

    Order and payload come from different places, and each is authoritative for
    its half:

      LangGraph returns ToolMessages in EMISSION order -- the order the model
      listed the calls, which mirrors how the student phrased the question
      ("how do I run it, and what does it mean"). It executes them
      concurrently, so completion order is a race and useless for display.

      The artifacts hold the chain payloads, which are far too large to
      serialise back through a ToolMessage and would be fed to the router for
      no reason.

    So: walk the messages for order, look each one up by step_id for content.
    A tool call that failed validation has no parseable result and no step, and
    simply does not appear.
    """
    steps: List[ToolStep] = []
    seen = set()
    for item in tool_results:
        step = artifacts.steps.get(item.step_id)
        if step is not None and step.step_id not in seen:
            seen.add(step.step_id)
            steps.append(step)
    # Correlation failed entirely (older payloads, or every ToolMessage errored)
    # -- fall back to completion order rather than showing the student nothing.
    if not steps:
        return artifacts.ordered_steps
    for step in artifacts.ordered_steps:
        if step.step_id not in seen:
            steps.append(step)
    return steps


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


def _format_tool_args(args: Dict[str, Any]) -> str:
    """The router's arguments, short enough for a status line."""
    parts = []
    for key, value in (args or {}).items():
        if value in ("", 0, None, [], {}):
            continue
        text = str(value).replace("\n", " ")
        if len(text) > 48:
            text = text[:45] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _announce(
    state: Dict[str, Any], progress: ProgressReporter, announced: set, rounds: List[int]
) -> None:
    """Turn one graph checkpoint into status updates.

    Called from the stream loop, which is the UI thread -- so these paint
    immediately, unlike the lines the tools raise from ToolNode's workers. That
    matters most here: naming the chosen tools BEFORE the tools node runs is
    what puts an honest label over the retrieval the student is waiting on, and
    gives them a chance to spot a misroute before reading a wrong answer.
    """
    messages = state.get("messages") or []
    if not messages:
        return
    last = messages[-1]
    key = getattr(last, "id", None) or id(last)
    if key in announced:
        return
    announced.add(key)

    if isinstance(last, AIMessage):
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            # No tool follow-up. app.py sets the writing label when the answer
            # is shown or when the tutoring chain emits its first token.
            return
        names = [
            (call.get("name") if isinstance(call, dict) else getattr(call, "name", ""))
            for call in calls
        ]
        names = [name for name in names if name]
        if rounds[0]:
            progress.emit(label=PHASE_RETRY, detail="Retrying with corrected arguments")
        rounds[0] += 1
        progress.emit(tools=names)
        for call, name in zip(calls, names):
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            shown = _format_tool_args(args)
            # Tool names and raw router arguments are machinery: the student
            # gets the plain-language line each tool writes for itself.
            progress.emit(
                detail=f"Chose {name}" + (f" ({shown})" if shown else ""), debug=True
            )


def run_ta_turn(
    *,
    agent,
    query: str,
    chat_history,
    artifacts: TurnArtifacts,
    memory_window: int = DEFAULT_MEMORY_WINDOW,
    progress: Optional[ProgressReporter] = None,
    # One round of tool calls costs 3 steps (agent -> tools -> agent), so a
    # limit of 4 left no budget for a second round. When a tool call failed
    # validation, LangGraph fed the model its own "Please fix the error and try
    # again" message -- and then had nowhere to put the retry, so a recoverable
    # argument error became silent, permanent data loss. 8 allows two rounds
    # plus margin, and the router prefers parallel calls anyway, so this rarely
    # costs an extra LLM call in practice.
    recursion_limit: int = 8,
) -> Dict[str, Any]:
    progress = progress or ProgressReporter()
    prior_messages = _history_to_messages(chat_history, max_messages=memory_window)
    started = time.perf_counter()

    # Streamed rather than invoked purely for the progress channel: `values`
    # hands us the full state after every node, so the router's decision is
    # readable the moment it is made instead of after the whole turn. The final
    # chunk is exactly what invoke() would have returned.
    progress.emit(label=PHASE_ROUTING)
    result: Dict[str, Any] = {}
    announced: set = set()
    rounds = [0]
    for chunk in agent.stream(
        {"messages": prior_messages + [HumanMessage(content=query)]},
        config={"recursion_limit": recursion_limit},
        stream_mode="values",
    ):
        result = chunk
        # Anything a tool raised from a worker thread lands here, at the first
        # moment it can actually be painted.
        progress.flush()
        _announce(chunk, progress, announced, rounds)
    progress.flush()

    router_ms = int((time.perf_counter() - started) * 1000)
    messages = result.get("messages", [])
    tool_results = _collect_tool_results(messages)
    tool_calls = _extract_tool_calls(messages)
    tools_used = [item["name"] for item in tool_calls] or list(artifacts.tools_used)

    steps = _ordered_steps(tool_results, artifacts)
    answerable = [step for step in steps if step.produced_answer]

    # Nothing prepared an answer -- either the router picked no tool, or every
    # call failed. Its own words are then the only thing to show.
    answer = "" if answerable else _final_answer(messages, tool_results)

    # These aggregates exist for the analytics event and the follow-up chips.
    # The UI reads the per-step values instead, because that is the whole point
    # of the refactor: one badge per section, each naming the tool that wrote it.
    practice_topic = next((s.practice_topic for s in steps if s.practice_topic), "")
    retrieval_debug = [row for step in steps for row in step.retrieval_debug]
    sources = [label for step in steps for label in step.sources]
    retrieval_quality = next((s.retrieval_quality for s in answerable if s.retrieval_quality), "")

    # A turn only counts as abstained when EVERY section the student will read
    # is a refusal. One empty lookup alongside a good answer is not a failure.
    abstained = bool(answerable) and all(step.abstained for step in answerable)
    if not answerable:
        abstained = any(step.abstained for step in steps)

    # Names the tool that wrote the FIRST section, which is what the student
    # reads first. Previously tools_used[0] -- emission order -- which could
    # name a tool whose work never reached the screen at all.
    route_label = answerable[0].tool_name if answerable else (
        tools_used[0] if tools_used else "agent_direct"
    )

    # The router's own words. Normally a throwaway acknowledgement, but when it
    # picks no tool this IS the answer, and when it picks a wrong tool this is
    # usually where the reason shows up.
    router_text = ""
    for message in messages:
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            if message.content.strip():
                router_text = message.content.strip()

    return {
        "steps": steps,
        "answerable_steps": answerable,
        "answer": answer,
        "tools_used": tools_used,
        "tool_calls": tool_calls,
        "route_label": route_label,
        "sources": sources,
        "retrieval_debug": retrieval_debug,
        "practice_topic": practice_topic,
        "abstained": abstained,
        "retrieval_quality": retrieval_quality,
        "messages": messages,
        "router_ms": router_ms,
        "router_text": router_text,
        "trace": [step.trace for step in steps if step.trace],
    }
