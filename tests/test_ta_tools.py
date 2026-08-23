"""What each tool puts in its chain payload, exercised without an LLM or an
index. The tools are plain closures; the payloads they prepare are the contract
with chains_lcel, so this is where a renamed key or a dropped field shows up."""

import json

from langchain_core.messages import AIMessage, HumanMessage

from utils import practice
from utils.ta_tools import (
    NO_HELD_QUESTION,
    TurnArtifacts,
    build_ta_tools,
    parse_tool_message_content,
)

ATTACHED = (
    "--- Attached file: attempt.txt ---\n"
    "The slope of 2.1 means price rises $2.10 per extra square foot."
)


def _tools(**kwargs):
    artifacts = TurnArtifacts()
    tools = build_ta_tools(
        contents_db=None,
        documents_db=None,
        chains_dict={},
        chat_history=kwargs.pop("chat_history", []),
        response_mode="Direct answer",
        artifacts=artifacts,
        course_context=kwargs.pop("course_context", "COURSE: test"),
        **kwargs,
    )
    return {t.name: t for t in tools}, artifacts


def _only_step(artifacts):
    assert len(artifacts.steps) == 1
    return next(iter(artifacts.steps.values()))


def test_tool_receipt_round_trips_through_the_parser():
    tools, artifacts = _tools()
    raw = tools["answer_course_facts"].invoke({"query": "When is the quiz?"})
    receipt = parse_tool_message_content(raw)
    step = _only_step(artifacts)
    assert receipt.tool_name == "answer_course_facts"
    assert receipt.stream_ready and receipt.step_id == step.step_id
    assert step.stream_spec.chain_key == "facts_chain"
    assert step.stream_spec.payload["course_context"] == "COURSE: test"
    # Not our JSON -> failure signal for the graph's retry edge.
    assert parse_tool_message_content("Error invoking tool") is None


def test_facts_tool_abstains_without_a_snapshot():
    tools, artifacts = _tools(course_context="")
    tools["answer_course_facts"].invoke({"query": "When is the quiz?"})
    step = _only_step(artifacts)
    assert step.abstained and step.stream_spec is None
    assert "check Canvas" in step.static_answer


def test_check_attempt_appends_attachment_the_router_did_not_copy():
    tools, artifacts = _tools(attachment_text=ATTACHED)
    tools["check_attempt"].invoke({"attempt_text": "Here is my interpretation.", "topic": "Regression"})
    payload = _only_step(artifacts).stream_spec.payload
    assert payload["attempt_text"].startswith("Here is my interpretation.")
    assert payload["attempt_text"].count("price rises $2.10") == 1


def test_check_attempt_does_not_double_an_attachment_the_router_copied():
    tools, artifacts = _tools(attachment_text=ATTACHED)
    # The router copied the body, dropped the marker line, and reflowed it.
    copied = "The slope of 2.1 means price rises $2.10  per extra square foot."
    tools["check_attempt"].invoke({"attempt_text": copied})
    payload = _only_step(artifacts).stream_spec.payload
    assert payload["attempt_text"].count("price rises") == 1


def test_check_attempt_replaces_a_partial_copy_with_the_whole_attachment():
    # Observed live: the router copied 882 of 1,925 characters and stopped.
    # Grading the fragment would mark the student down for work they did.
    long_body = "Sentence number %d about the Eastville price distribution. "
    attachment = "--- Attached file: attempt.txt ---\n" + "".join(long_body % i for i in range(30))
    tools, artifacts = _tools(attachment_text=attachment)
    partial = "".join(long_body % i for i in range(12)).strip()
    tools["check_attempt"].invoke({"attempt_text": partial})
    payload = _only_step(artifacts).stream_spec.payload
    assert "Sentence number 29" in payload["attempt_text"]
    assert payload["attempt_text"].count("Sentence number 3 ") == 1


def test_check_attempt_keeps_typed_words_and_adds_the_attachment():
    tools, artifacts = _tools(attachment_text=ATTACHED)
    tools["check_attempt"].invoke({"attempt_text": "I think the sign is right but not sure about units."})
    payload = _only_step(artifacts).stream_spec.payload
    assert payload["attempt_text"].startswith("I think the sign is right")
    assert "price rises" in payload["attempt_text"]


def test_check_attempt_with_only_an_attachment_grades_it():
    tools, artifacts = _tools(attachment_text=ATTACHED)
    tools["check_attempt"].invoke({"attempt_text": ""})
    step = _only_step(artifacts)
    assert step.stream_spec is not None
    assert "price rises" in step.stream_spec.payload["attempt_text"]


def test_check_attempt_asks_for_text_when_there_is_nothing_to_grade():
    tools, artifacts = _tools()
    tools["check_attempt"].invoke({"attempt_text": ""})
    step = _only_step(artifacts)
    assert step.stream_spec is None and "paste your attempt" in step.static_answer.lower()


def test_check_attempt_uses_held_question_or_the_placeholder():
    session = practice.start("Q: interpret the slope", "Regression")
    tools, artifacts = _tools(practice_session=session)
    tools["check_attempt"].invoke({"attempt_text": "slope means..."})
    assert _only_step(artifacts).stream_spec.payload["question"] == "Q: interpret the slope"

    tools, artifacts = _tools()
    tools["check_attempt"].invoke({"attempt_text": "slope means..."})
    assert _only_step(artifacts).stream_spec.payload["question"] == NO_HELD_QUESTION


def test_check_attempt_routes_screenshots_to_the_vision_chain():
    images = [{"name": "out.png", "data_url": "data:image/png;base64,AAAA"}]
    tools, artifacts = _tools(images=images)
    tools["check_attempt"].invoke({"attempt_text": ""})
    step = _only_step(artifacts)
    assert step.stream_spec.chain_key == "check_chain_vision"
    assert step.stream_spec.payload["images"] == images
    assert "screenshot" in step.stream_spec.payload["attempt_text"]


def test_coach_practice_refuses_with_no_open_question():
    tools, artifacts = _tools()
    tools["coach_practice"].invoke({"query": "I'm stuck", "request": "hint"})
    step = _only_step(artifacts)
    assert step.stream_spec is None
    assert "don't have a practice question open" in step.static_answer


def test_coach_practice_escalates_after_max_hints():
    session = practice.start("Q: fold back the tree", "Decision analysis")
    for _ in range(practice.MAX_HINTS):
        practice.record_hint(session)
    tools, artifacts = _tools(practice_session=session)
    tools["coach_practice"].invoke({"query": "another hint please", "request": "hint"})
    payload = _only_step(artifacts).stream_spec.payload
    assert payload["request"] == "worked_step"
    assert "Q: fold back the tree" in payload["question_block"]


def test_generate_practice_carries_previous_question_and_flags_misroute():
    session = practice.start("Q: old question", "Regression")
    practice.record_hint(session)
    history = [HumanMessage("explain multicollinearity"), AIMessage("...")]
    tools, artifacts = _tools(practice_session=session, chat_history=history)
    tools["generate_practice"].invoke({"topic": "", "difficulty": "HARDER"})
    step = _only_step(artifacts)
    payload = step.stream_spec.payload
    assert payload["topic"] == "Multiple regression"   # inferred from history
    assert payload["difficulty"] == "harder"
    assert payload["previous_question"] == "Q: old question"
    assert step.trace["probable_misroute"] is True


def test_course_documents_needs_a_topic_or_a_filter():
    tools, artifacts = _tools()
    raw = tools["answer_course_documents"].invoke({"query": ""})
    assert json.loads(raw)["abstained"] is True
    assert "topic or a date" in _only_step(artifacts).static_answer


# ---- compound turns -----------------------------------------------------------
from utils.ta_tools import COMPOUND_SECTION_WORDS, ToolStep, StreamSpec, annotate_compound_turn


def _streamed(tool, covers, chain="x"):
    step = ToolStep(step_id=tool, tool_name=tool, covers=covers)
    step.stream_spec = StreamSpec(chain_key=chain, payload={})
    return step


def test_single_section_is_not_a_compound_turn():
    only = _streamed("answer_concept", "what R-squared means")
    assert annotate_compound_turn([only]) == 0
    assert "turn_context" not in only.stream_spec.payload


def test_a_refusal_next_to_one_answer_is_not_compound():
    refusal = ToolStep(step_id="r", tool_name="answer_course_documents", static_answer="Not found.")
    answer = _streamed("answer_concept", "what R-squared means")
    assert annotate_compound_turn([refusal, answer]) == 0


def test_compound_turn_tells_each_part_about_the_others():
    jmp = _streamed("answer_software", "how to run a regression in JMP")
    r2 = _streamed("answer_concept", "how to interpret R-squared")
    assert annotate_compound_turn([jmp, r2]) == 2

    first = jmp.stream_spec.payload["turn_context"]
    second = r2.stream_spec.payload["turn_context"]
    assert first.startswith("THIS IS PART 1 OF 2")
    assert "Part 1 (you): how to run a regression in JMP" in first
    assert "Part 2 (written separately): how to interpret R-squared" in first
    assert "not the last part" in first and "No closing question" in first
    assert second.startswith("THIS IS PART 2 OF 2")
    assert "Part 1 (written separately): how to run a regression in JMP" in second
    assert "You are the last part" in second
    assert f"under {COMPOUND_SECTION_WORDS} words" in first


def test_every_tool_records_what_its_section_covers():
    session = practice.start("Q: fold back the tree", "Decision basics")
    tools, artifacts = _tools(practice_session=session, attachment_text="")
    tools["answer_course_facts"].invoke({"query": "When is quiz 1 due?"})
    tools["answer_software"].invoke({"query": "How do I open Fit Model?"})
    tools["generate_practice"].invoke({"topic": "Decision basics"})
    tools["coach_practice"].invoke({"query": "stuck", "request": "hint"})
    tools["check_attempt"].invoke({"attempt_text": "EV = 0.4*100", "topic": "Decision basics"})
    covers = {s.tool_name: s.covers for s in artifacts.steps.values()}
    assert covers["answer_course_facts"] == "When is quiz 1 due?"
    assert covers["answer_software"] == "How do I open Fit Model?"
    assert "practice question on Decision basics" in covers["generate_practice"]
    assert "practice question on screen" in covers["coach_practice"]
    assert "attempt" in covers["check_attempt"]
