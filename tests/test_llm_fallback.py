"""ModelWithFallback must fall back on STREAMING failures, not only invoke().

Every tutoring chain streams. BaseChatModel.stream is a generator function, so
`primary.stream(...)` cannot raise at call time -- the request is sent on the
first next(). A try/except around the call alone never catches anything, which
is how the main tutor's fallback was dead for every streamed answer.
"""

from typing import Any, Iterator, List, Optional

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from utils.llm_models import ModelWithFallback


class _Scripted(BaseChatModel):
    """Yields `text` one word at a time, or raises `error` on the first token."""

    text: str = ""
    error: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: List[BaseMessage], stop=None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None,
                  **kwargs: Any) -> ChatResult:
        if self.error:
            raise RuntimeError(self.error)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.text))])

    def _stream(self, messages: List[BaseMessage], stop=None,
                run_manager: Optional[CallbackManagerForLLMRun] = None,
                **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        if self.error:
            raise RuntimeError(self.error)
        for word in self.text.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def _collect(stream) -> str:
    return "".join(chunk.content for chunk in stream).strip()


def test_stream_uses_primary_when_it_works():
    model = ModelWithFallback(
        primary=_Scripted(text="primary answer"),
        fallback=_Scripted(text="fallback answer"),
        verbose=False,
    )
    assert _collect(model.stream("q")) == "primary answer"


def test_stream_falls_back_when_primary_fails_on_first_token():
    model = ModelWithFallback(
        primary=_Scripted(error="503 from provider"),
        fallback=_Scripted(text="fallback answer"),
        verbose=False,
    )
    assert _collect(model.stream("q")) == "fallback answer"


def test_stream_raises_when_both_fail():
    model = ModelWithFallback(
        primary=_Scripted(error="down"),
        fallback=_Scripted(error="also down"),
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="also down"):
        _collect(model.stream("q"))


def test_invoke_still_falls_back():
    model = ModelWithFallback(
        primary=_Scripted(error="down"),
        fallback=_Scripted(text="fallback answer"),
        verbose=False,
    )
    assert model.invoke("q").content == "fallback answer"
