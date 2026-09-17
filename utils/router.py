"""The router: one model call that picks tools, then the tools run in order.

A turn is two model phases with Python in between:

    student text  ->  router (tool calling)  ->  each tool PREPARES a chain
    payload on its ToolStep and returns a receipt  ->  app.py streams the
    prepared chains, one section per tool call, in the order the router asked

The router is a dispatcher, not a writer. It never sees retrieved text -- a
tool returns only a receipt ({answer, abstained, tool_name, stream_ready,
step_id}) -- so there is no second router pass that previews or rewrites the
answer. When the router calls no tool (greetings, "what can you do"), its own
reply is the answer.

This used to be a LangGraph state machine. The graph only ever ran
agent -> tools -> END, with one extra edge to let the router retry a tool call
that failed validation; the loop below is the same control flow written out,
minus the retry and minus a thread pool the tools did not need.
"""

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from utils.ta_tools import (
    ToolExecutionResult,
    ToolStep,
    TurnArtifacts,
    annotate_compound_turn,
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
- generate_practice: when the student wants a NEW practice question
- coach_practice: the student is stuck on the practice question already on their
  screen -- a hint, "what is this asking", "I don't understand"
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
5) Practice, coaching and attempts:
   - generate_practice when the student wants a NEW question, or explicitly asks
     for another / a different / a harder / an easier one. Set difficulty to
     'easier', 'same' or 'harder'.
   - coach_practice whenever a practice question is already on screen and the
     student is stuck: "hint", "I'm stuck", "I don't understand", "this is
     confusing", "where do I start", "what is it asking". Set request='hint',
     'clarify' or 'worked_step'. This never replaces the question on screen.
   - check_attempt when they want feedback on work they wrote. Pass what they
     typed in attempt_text. Do NOT copy attached file blocks into it -- the
     attachment is handed to the checker automatically.
   - Being stuck is NOT a request for a new question. If a practice question is
     open and the student expresses difficulty rather than asking for a
     different question, that is coach_practice, never generate_practice --
     generating replaces the question they are working on and loses their place.
6) Attachments:
   - Blocks marked "--- Attached file: ... ---" are the student's own work or
     data, not course material. Route to check_attempt when they want it
     reviewed; the checker receives the attachment itself, so do not copy it.
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


@dataclass
class Router:
    """The tool-calling model plus the tools it may call, built fresh per turn."""

    model: Any  # agent_llm.bind_tools(tools)
    tools: Dict[str, Any]  # name -> StructuredTool
    system_prompt: str


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
    system_prompt: Optional[str] = None,
    memory_window: int = DEFAULT_MEMORY_WINDOW,
    images: Optional[List[Dict[str, str]]] = None,
    practice_session: Optional[Dict[str, Any]] = None,
    attachment_text: str = "",
) -> Router:
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
        memory_window=memory_window,
        # Handed to the tools, never to the router: `agent_llm` only has to
        # choose a route, and the query already carries a one-line marker
        # saying a screenshot is present and readable.
        images=images,
        # The practice question on screen. Read by generate_practice (so a
        # "harder" variant escalates instead of repeating), check_attempt (so
        # grading sees the real question) and coach_practice (which has nothing
        # to do without it).
        practice_session=practice_session,
        # Decoded text/PDF attachments, for check_attempt. Same reasoning as
        # images: the router sees the content in its query and only has to
        # route; it does not have to copy it into a tool call.
        attachment_text=attachment_text,
    )
    return Router(
        model=agent_llm.bind_tools(tools),
        tools={tool.name: tool for tool in tools},
        system_prompt=system_prompt or _agent_system_prompt(),
    )


def _ordered_steps(
    tool_results: List[ToolExecutionResult], artifacts: TurnArtifacts
) -> List[ToolStep]:
    """The prepared steps, in the order the router asked for them.

    Each receipt carries the step_id of the ToolStep holding its chain payload
    (payloads are too large to pass back through the receipt). A tool call that
    raised has no receipt and no step, and simply does not appear.
    """
    steps: List[ToolStep] = []
    seen = set()
    for item in tool_results:
        step = artifacts.steps.get(item.step_id)
        if step is not None and step.step_id not in seen:
            seen.add(step.step_id)
            steps.append(step)
    for step in artifacts.ordered_steps:
        if step.step_id not in seen:
            steps.append(step)
    return steps


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
        return "\n".join(p for p in parts if p).strip()
    return ""


def run_ta_turn(
    *,
    agent: Router,
    query: str,
    chat_history,
    artifacts: TurnArtifacts,
    memory_window: int = DEFAULT_MEMORY_WINDOW,
    on_route: Optional[Callable[[List[str]], None]] = None,
) -> Dict[str, Any]:
    """Route one student message and run the tools it names.

    `on_route` is called with the chosen tool names the moment the router has
    answered, before any tool runs -- that is the one honest moment to put a
    label like "Looking up the course schedule" over the wait that follows.
    """
    started = time.perf_counter()
    messages = (
        [SystemMessage(content=agent.system_prompt)]
        + _history_to_messages(chat_history, max_messages=memory_window)
        + [HumanMessage(content=query)]
    )
    reply: AIMessage = agent.model.invoke(messages)
    router_ms = int((time.perf_counter() - started) * 1000)

    tool_calls = [
        {"name": call.get("name", ""), "args": call.get("args") or {}}
        for call in (getattr(reply, "tool_calls", None) or [])
        if call.get("name")
    ]
    if on_route and tool_calls:
        on_route([call["name"] for call in tool_calls])

    tool_results: List[ToolExecutionResult] = []
    for call in tool_calls:
        tool = agent.tools.get(call["name"])
        if tool is None:
            print(f"[router] unknown tool {call['name']!r} -- skipped")
            continue
        try:
            receipt = tool.invoke(call["args"])
        except Exception:
            # A tool that raises loses its section; the rest of the turn goes on.
            print(f"[router] {call['name']} failed:\n{traceback.format_exc()}")
            continue
        parsed = parse_tool_message_content(receipt)
        if parsed is not None:
            tool_results.append(parsed)

    steps = _ordered_steps(tool_results, artifacts)
    answerable = [step for step in steps if step.produced_answer]
    # Two or more sections: tell each what the others cover, so they read as
    # parts of one answer rather than two full replies stapled together.
    annotate_compound_turn(steps)

    # Nothing prepared an answer -- the router called no tool, or every call
    # failed. Its own words are then the only thing to show.
    router_text = _message_text(reply)
    answer = "" if answerable else (router_text or "I could not generate a response for that request.")

    # Aggregates for the analytics event and the follow-up chips. The UI reads
    # the per-step values instead: one badge per section, naming its tool.
    practice_topic = next((s.practice_topic for s in steps if s.practice_topic), "")
    retrieval_debug = [row for step in steps for row in step.retrieval_debug]
    sources = [label for step in steps for label in step.sources]
    retrieval_quality = next((s.retrieval_quality for s in answerable if s.retrieval_quality), "")

    # A turn only counts as abstained when EVERY section the student will read
    # is a refusal. One empty lookup alongside a good answer is not a failure.
    abstained = bool(answerable) and all(step.abstained for step in answerable)
    if not answerable:
        abstained = any(step.abstained for step in steps)

    tools_used = [call["name"] for call in tool_calls]
    route_label = answerable[0].tool_name if answerable else (
        tools_used[0] if tools_used else "agent_direct"
    )

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
        "router_ms": router_ms,
        "router_text": router_text,
        "trace": [step.trace for step in steps if step.trace],
    }
