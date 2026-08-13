"""What the agent is doing, reported while it is still doing it.

Most of a turn's wall clock is spent before the student sees a single token:
the router LLM call, then retrieval, then the first chunk of the tutoring
chain. The status bar used to sit on one hardcoded "Looking up course
materials..." through all of it, so a slow retrieval and a misroute looked
identical. This module is the channel that carries each step out as it starts.

Two rules make it safe to call from anywhere in the turn:

  Delivery is thread-aware. LangGraph's ToolNode runs EVERY tool call through
  a thread pool (see ToolNode._func -- even a single call goes through
  executor.map), and Streamlit silently drops widget writes from threads that
  carry no ScriptRunContext. So an event raised inside a tool cannot paint
  itself. Those are buffered instead, and `flush()` -- called from the graph
  stream loop, which is on the owning thread by definition -- delivers them at
  the next node boundary.

  A failing sink never fails the turn. Progress is commentary; if the status
  widget is gone or misbehaves, the answer still has to arrive.

The events carry no Streamlit and no phrasing decisions for tools: a tool
event says what happened, and `tools=` events name the route and let the UI
layer (ui.working_label) choose the words, so the live label and the finished
badge cannot drift apart.
"""

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple


# Phases that belong to the turn itself rather than to any one tool.
PHASE_ROUTING = "Reading your question"
PHASE_RETRY = "Correcting the lookup"
PHASE_WRITING = "Generating your answer"
PHASE_RETRIEVING = "Retrieving course materials"
PHASE_DONE = "Answer complete"
PHASE_FAILED = "Course lookup failed"


@dataclass(frozen=True)
class ProgressEvent:
    """One thing the agent did.

    label:  new status-bar text. Empty means "leave the label alone".
    tools:  route labels whose phrasing the UI should turn into a label.
    detail: one line for the status log. Never changes the label.
    state:  running | complete | error, passed straight to st.status.
    debug:  instructor-only. Machinery a student has no use for -- tool names,
            raw router arguments. Shown only when diagnostics are on, and never
            kept in the finished turn's record, where the diagnostics expander
            already reports the same calls in full.
    """

    label: str = ""
    detail: str = ""
    state: str = "running"
    tools: Tuple[str, ...] = ()
    debug: bool = False


class ProgressReporter:
    """Collects progress events and replays them on the thread that owns the UI."""

    def __init__(self, sink: Optional[Callable[[ProgressEvent], None]] = None):
        self._sink = sink
        # Whoever built the reporter owns the UI: in the app that is the
        # Streamlit script thread, in a script or a test it is main.
        self._owner = threading.current_thread()
        self._lock = threading.Lock()
        self._pending: List[ProgressEvent] = []
        self.events: List[ProgressEvent] = []

    def emit(
        self,
        label: str = "",
        *,
        detail: str = "",
        state: str = "running",
        tools: Sequence[str] = (),
        debug: bool = False,
    ) -> None:
        event = ProgressEvent(
            label=label, detail=detail, state=state, tools=tuple(tools), debug=debug
        )
        with self._lock:
            self.events.append(event)
            off_thread = threading.current_thread() is not self._owner
            if off_thread:
                self._pending.append(event)
        if not off_thread:
            self._deliver(event)

    def flush(self) -> None:
        """Deliver everything raised off-thread. Call from the owning thread."""
        with self._lock:
            pending, self._pending = self._pending, []
        for event in pending:
            self._deliver(event)

    def _deliver(self, event: ProgressEvent) -> None:
        if self._sink is None:
            return
        try:
            self._sink(event)
        except Exception:
            # A progress line is never worth losing an answer over.
            pass

    @property
    def log(self) -> List[str]:
        """The student-facing detail lines, in order. Debug lines are live-only."""
        return [e.detail for e in self.events if e.detail and not e.debug]

    def summary(self) -> dict:
        """The finished turn's status, in a form the transcript can redraw.

        The live st.status is destroyed by the rerun that ends the turn, so
        without this the record of how an answer was built lasts only as long
        as the student takes to read it. Stored per message instead, and drawn
        back collapsed above the answer.
        """
        label = ""
        state = "complete"
        for event in self.events:
            if event.label:
                label = event.label
                state = event.state
        return {
            # A finished turn never spins: a "running" state here means the
            # turn ended without a terminal event, not that work continues.
            "label": label or PHASE_DONE,
            "state": "complete" if state == "running" else state,
            "lines": self.log,
        }
