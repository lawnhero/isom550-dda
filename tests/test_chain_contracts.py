"""Every tool's payload must satisfy its chain's template.

The tools and the chains live in different modules and agree only by key
name. A renamed payload key or a new {placeholder} in a template fails at
stream time, in front of a student, as a KeyError. This builds every chain on
a fake model, runs every tool against a stub index, and streams the payload
each tool prepared through the chain it named.
"""

from typing import Any, Iterator, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

import utils.chains_lcel as chains
from utils import practice
from utils.ta_tools import TurnArtifacts, build_ta_tools


class _Echo(BaseChatModel):
    """Returns the rendered prompt, so a test can assert on what the model saw."""

    @property
    def _llm_type(self) -> str:
        return "echo"

    @staticmethod
    def _text(messages: List[BaseMessage]) -> str:
        parts = []
        for m in messages:
            parts.append(m.content if isinstance(m.content, str)
                         else " ".join(p.get("text", "") for p in m.content if isinstance(p, dict)))
        return "\n".join(parts)

    def _generate(self, messages, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._text(messages)))])

    def _stream(self, messages, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content=self._text(messages)))


class _StubIndex:
    """Enough of langchain_chroma.Chroma for search_concepts / search_documents."""

    def __init__(self, docs, distance=1.0):
        self._docs = docs
        self._distance = distance
        self._persist_directory = None
        self._collection = self

    def count(self):
        return len(self._docs)

    def similarity_search_with_score(self, query, k=4, filter=None):
        return [(d, self._distance) for d in self._docs[:k]]

    def get(self, where=None, limit=None):
        return {
            "documents": [d.page_content for d in self._docs],
            "metadatas": [d.metadata for d in self._docs],
        }


CONCEPTS = [
    Document(
        page_content="Interpreting R-squared\n\nR-squared is the share of variation explained.",
        metadata={"concept_id": "r-squared", "title": "Interpreting R-squared",
                  "module": "simple-regression", "body": "R-squared is the share of variation explained.",
                  "managerial_phrasing": "Say what share of the variation the model explains.",
                  "common_mistake": "Treating a high R-squared as proof of causation."},
    ),
]
DOCUMENTS = [
    Document(
        page_content=(
            "Assignment: Eastville Part 1\n\nSummarise the distribution of house prices "
            "in the Eastville data: centre, spread and shape, with one chart. (10 points)"
        ),
        metadata={"doc_type": "assignment", "title": "Eastville Part 1",
                  "url": "https://canvas.example/a/1", "ymd": 20260902, "chunk": 0, "n_chunks": 1},
    ),
]


@pytest.fixture
def all_chains():
    return chains.get_all_chains(_Echo(), _Echo(), vision_llm=_Echo())


@pytest.fixture
def tools():
    artifacts = TurnArtifacts()
    session = practice.start("Q: interpret an R-squared of 0.62", "Regression")
    built = build_ta_tools(
        contents_db=_StubIndex(CONCEPTS, distance=1.1),
        documents_db=_StubIndex(DOCUMENTS, distance=1.1),
        chains_dict={},
        chat_history=[HumanMessage("what does r-squared mean"), AIMessage("...")],
        response_mode="Teach me step-by-step",
        artifacts=artifacts,
        course_context="COURSE: ISOM 550 (Fall 2026)\nUPCOMING DEADLINES\n  - Quiz 1 — due Wed Sep 2",
        software_context="SOFTWARE THIS COURSE USES\n  JMP: JMP 18",
        course_span=(20260806, 20261123),
        practice_session=session,
        attachment_text="--- Attached file: work.txt ---\nR-squared of 0.62 means 62% explained.",
    )
    return {t.name: t for t in built}, artifacts


CALLS = [
    ("answer_course_facts", {"query": "When is quiz 1 due?"}, "facts_chain"),
    ("answer_software", {"query": "How do I fit a line in JMP?"}, "software_chain"),
    ("answer_course_documents", {"query": "Eastville", "doc_type": "assignment"}, "doc_chain"),
    ("answer_course_documents", {"query": "", "days_back": 30}, "doc_chain"),
    ("answer_concept", {"query": "what does r-squared mean", "module": "simple-regression"}, "concept_chain"),
    ("generate_practice", {"topic": "Regression", "difficulty": "harder"}, "practice_chain"),
    ("coach_practice", {"query": "I'm stuck", "request": "clarify"}, "coach_chain"),
    ("check_attempt", {"attempt_text": "62% of variation is explained"}, "check_chain"),
]


@pytest.mark.parametrize("tool_name, args, chain_key", CALLS)
def test_tool_payload_renders_through_its_chain(tools, all_chains, tool_name, args, chain_key):
    built, artifacts = tools
    built[tool_name].invoke(args)
    step = next(iter(artifacts.steps.values()))
    assert step.stream_spec is not None, step.static_answer
    assert step.stream_spec.chain_key == chain_key
    rendered = "".join(all_chains[chain_key].stream(step.stream_spec.payload))
    assert "Dayton" in rendered
    assert "{" not in rendered.replace("{}", ""), "unrendered template variable"


def test_concept_payload_carries_phrasing_and_mistake(tools, all_chains):
    built, artifacts = tools
    built["answer_concept"].invoke({"query": "what does r-squared mean", "module": "simple-regression"})
    step = next(iter(artifacts.steps.values()))
    rendered = "".join(all_chains["concept_chain"].stream(step.stream_spec.payload))
    assert "How to phrase it:" in rendered
    assert "Common student mistake:" in rendered
    assert step.retrieval_quality == "strong"


def test_document_payload_carries_the_canvas_link(tools, all_chains):
    built, artifacts = tools
    built["answer_course_documents"].invoke({"query": "Eastville", "doc_type": "assignment"})
    step = next(iter(artifacts.steps.values()))
    rendered = "".join(all_chains["doc_chain"].stream(step.stream_spec.payload))
    assert "Source: Eastville Part 1 — https://canvas.example/a/1" in rendered
    assert step.sources == ["Eastville Part 1"]


def test_filter_only_documents_question_is_synthesised(tools, all_chains):
    built, artifacts = tools
    built["answer_course_documents"].invoke({"query": "", "days_back": 30, "doc_type": "announcement"})
    step = next(iter(artifacts.steps.values()))
    assert step.stream_spec.payload["query"].startswith("Summarise what these class sessions")
    assert step.trace["mode"] == "filter"


def test_vision_build_prepends_the_vision_policy(all_chains):
    payload = {
        "topic": "Regression", "attempt_text": "see screenshot", "question": "",
        "chat_history": "No previous conversation.",
        "images": [{"name": "a.png", "data_url": "data:image/png;base64,AAAA"}],
    }
    rendered = "".join(all_chains["check_chain_vision"].stream(payload))
    assert rendered.startswith(chains.VISION_POLICY[:40])


def test_far_concept_abstains(all_chains):
    artifacts = TurnArtifacts()
    built = {t.name: t for t in build_ta_tools(
        contents_db=_StubIndex(CONCEPTS, distance=1.9), documents_db=None, chains_dict={},
        chat_history=[], response_mode="Direct answer", artifacts=artifacts,
    )}
    built["answer_concept"].invoke({"query": "What is a Kubernetes pod?", "module": ""})
    step = next(iter(artifacts.steps.values()))
    assert step.abstained and step.retrieval_quality == "none"


def test_concept_answer_names_the_retrieved_concept_as_practice_topic(tools):
    built, artifacts = tools
    built["answer_concept"].invoke({"query": "What does an R-squared of 0.62 mean?", "module": "simple-regression"})
    step = next(iter(artifacts.steps.values()))
    # Not the keyword-inferred "Regression": the concept that answered.
    assert step.practice_topic == "Interpreting R-squared"


def test_weak_concept_match_falls_back_to_the_hit_module():
    artifacts = TurnArtifacts()
    built = {t.name: t for t in build_ta_tools(
        contents_db=_StubIndex(CONCEPTS, distance=1.5), documents_db=None, chains_dict={},
        chat_history=[], response_mode="Direct answer", artifacts=artifacts,
    )}
    built["answer_concept"].invoke({"query": "what does r-squared mean", "module": ""})
    step = next(iter(artifacts.steps.values()))
    assert step.retrieval_quality == "weak"
    assert step.practice_topic == "Simple regression"


def test_pill_focus_grounds_inside_its_module(all_chains):
    class _Filtering(_StubIndex):
        def similarity_search_with_score(self, query, k=4, filter=None):
            assert filter == {"module": {"$eq": "simple-regression"}}
            return super().similarity_search_with_score(query, k, filter)

    artifacts = TurnArtifacts()
    built = {t.name: t for t in build_ta_tools(
        contents_db=_Filtering(CONCEPTS, distance=1.1), documents_db=None, chains_dict={},
        chat_history=[], response_mode="Direct answer", artifacts=artifacts,
    )}
    built["generate_practice"].invoke({"topic": "Simple regression: R-squared and model fit"})
    step = next(iter(artifacts.steps.values()))
    # 1.1 is over the unfiltered bar (1.0) but inside the in-module bar (1.2).
    assert step.trace["module"] == "simple-regression"
    assert step.trace["grounded_on"] == "Interpreting R-squared"


def test_practice_is_grounded_in_a_close_concept(all_chains):
    artifacts = TurnArtifacts()
    built = {t.name: t for t in build_ta_tools(
        contents_db=_StubIndex(CONCEPTS, distance=0.5), documents_db=None, chains_dict={},
        chat_history=[], response_mode="Direct answer", artifacts=artifacts,
    )}
    built["generate_practice"].invoke({"topic": "Interpreting R-squared"})
    step = next(iter(artifacts.steps.values()))
    payload = step.stream_spec.payload
    assert "Common student mistake:" in payload["concept_context"]
    assert step.sources == ["Interpreting R-squared"]
    assert step.trace["grounded_on"] == "Interpreting R-squared"
    rendered = "".join(all_chains["practice_chain"].stream(payload))
    assert "Treating a high R-squared as proof of causation" in rendered


def test_practice_is_not_grounded_on_a_loose_match():
    artifacts = TurnArtifacts()
    built = {t.name: t for t in build_ta_tools(
        contents_db=_StubIndex(CONCEPTS, distance=1.2), documents_db=None, chains_dict={},
        chat_history=[], response_mode="Direct answer", artifacts=artifacts,
    )}
    built["generate_practice"].invoke({"topic": "Regression"})
    step = next(iter(artifacts.steps.values()))
    assert step.sources == []
    assert step.stream_spec.payload["concept_context"].startswith("(none")
    assert step.trace["grounded_on"] == "(none)"


def test_compound_turn_context_reaches_both_chains(tools, all_chains):
    from utils.ta_tools import annotate_compound_turn

    built, artifacts = tools
    built["answer_software"].invoke({"query": "How do I fit a line in JMP?"})
    built["answer_concept"].invoke({"query": "what does r-squared mean", "module": "simple-regression"})
    steps = list(artifacts.steps.values())
    assert annotate_compound_turn(steps) == 2
    for step in steps:
        rendered = "".join(all_chains[step.stream_spec.chain_key].stream(step.stream_spec.payload))
        assert "THIS IS PART" in rendered
        assert "written separately" in rendered


def test_single_turn_renders_an_empty_context_slot(tools, all_chains):
    built, artifacts = tools
    built["answer_software"].invoke({"query": "How do I fit a line in JMP?"})
    step = next(iter(artifacts.steps.values()))
    rendered = "".join(all_chains["software_chain"].stream(step.stream_spec.payload))
    assert "THIS IS PART" not in rendered and "{turn_context}" not in rendered


def test_empty_module_filter_widens_instead_of_abstaining(all_chains):
    """A module the router names but the index cannot match (renamed since the
    build, or simply the wrong guess) must not hide a concept the index holds."""
    class _ModuleAware(_StubIndex):
        def similarity_search_with_score(self, query, k=4, filter=None):
            if filter == {"module": {"$eq": "hypothesis-testing"}}:
                return []   # nothing filed there
            return super().similarity_search_with_score(query, k, filter)

    artifacts = TurnArtifacts()
    built = {t.name: t for t in build_ta_tools(
        contents_db=_ModuleAware(CONCEPTS, distance=1.0), documents_db=None, chains_dict={},
        chat_history=[], response_mode="Direct answer", artifacts=artifacts,
    )}
    built["answer_concept"].invoke({"query": "what does a p-value of 0.03 mean", "module": "hypothesis-testing"})
    step = next(iter(artifacts.steps.values()))
    assert not step.abstained
    assert step.trace["widened_from_module"] == "hypothesis-testing"
    assert step.sources == ["Interpreting R-squared"]
