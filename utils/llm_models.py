import os

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from pydantic import Field

load_dotenv()

# Create a couple of Global Variables
TEMPERATURE = 0.1
MAX_TOKENS = 1024


class ModelWithFallback(BaseChatModel):
    """A wrapper around two LLM models that falls back to the second if the first fails.
    
    This class provides automatic fallback capabilities for any LangChain BaseChatModel.
    It attempts to use the primary model for all operations, and if that fails,
    it automatically falls back to the secondary model.
    
    Attributes:
        primary: The primary LLM to use for generation
        fallback: The fallback LLM to use when the primary fails
        verbose: Whether to print detailed logs about fallbacks
    """
    primary: BaseChatModel = Field(description="Primary model to use")
    fallback: BaseChatModel = Field(description="Fallback model to use when primary fails")
    verbose: bool = Field(default=True, description="Whether to print detailed logs about fallbacks")
    
    def _log_fallback(self, error: Exception, method_name: str) -> None:
        """Log fallback information if verbose is enabled"""
        if self.verbose:
            print(f"Primary model {method_name} failed with error: {error}. Falling back to backup model.")
    
    def _generate(self, *args, **kwargs):
        try:
            return self.primary._generate(*args, **kwargs)
        except Exception as e:
            self._log_fallback(e, "_generate")
            return self.fallback._generate(*args, **kwargs)

    async def _agenerate(self, *args, **kwargs):
        try:
            return await self.primary._agenerate(*args, **kwargs)
        except Exception as e:
            self._log_fallback(e, "_agenerate")
            return await self.fallback._agenerate(*args, **kwargs)

    def invoke(self, *args, **kwargs):
        try:
            return self.primary.invoke(*args, **kwargs)
        except Exception as e:
            self._log_fallback(e, "invoke")
            return self.fallback.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        try:
            return await self.primary.ainvoke(*args, **kwargs)
        except Exception as e:
            self._log_fallback(e, "ainvoke")
            return await self.fallback.ainvoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        # BaseChatModel.stream is a generator function: calling it builds a
        # lazy generator and cannot raise. The request is only sent on the
        # first next(), so the failure has to be caught there. Once a chunk has
        # been yielded the primary owns the answer -- switching models
        # mid-sentence would splice two different replies together.
        try:
            primary = self.primary.stream(*args, **kwargs)
            first = next(primary)
        except StopIteration:
            return
        except Exception as e:
            self._log_fallback(e, "stream")
            yield from self.fallback.stream(*args, **kwargs)
            return
        yield first
        yield from primary

    async def astream(self, *args, **kwargs):
        try:
            primary = self.primary.astream(*args, **kwargs)
            first = await primary.__anext__()
        except StopAsyncIteration:
            return
        except Exception as e:
            self._log_fallback(e, "astream")
            async for chunk in self.fallback.astream(*args, **kwargs):
                yield chunk
            return
        yield first
        async for chunk in primary:
            yield chunk

    @property
    def _llm_type(self) -> str:
        return f"ModelWithFallback({self.primary._llm_type}->{self.fallback._llm_type})"


def create_model_with_fallback(
    primary_model: BaseChatModel,
    fallback_model: BaseChatModel
) -> BaseChatModel:
    """Creates a wrapper around the primary model that falls back to a secondary model if the primary fails"""
    return ModelWithFallback(primary=primary_model, fallback=fallback_model)


# Router budget. A tool call is JSON the model has to write out, and a turn can
# carry two of them plus a verbatim `attempt_text`. 300 was enough for a query
# string and nothing else: a pasted paragraph truncated the call mid-JSON and
# the turn failed with nothing to retry. Attached files no longer travel
# through this channel (see ta_tools.check_attempt), so this only has to cover
# what the student typed.
ROUTER_MAX_TOKENS = 900

openai_gpt56_luna = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-5.6-luna",
    max_tokens=ROUTER_MAX_TOKENS,
    reasoning_effort="none",  # required for /v1/chat/completions + tool calling
)

# Back-compat alias for agent dispatch and haiku/deepseek fallbacks.
openai_gpt4o_mini = openai_gpt56_luna

# Vision route. Every screenshot turn is pinned here rather than to the usual
# tutoring model: the instance above is the dispatch build with a tool-calling
# budget, and a "check my JMP output" answer needs the full budget plus room
# to transcribe what it read back to the student first.
openai_gpt56_luna_vision = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-5.6-luna",
    max_tokens=MAX_TOKENS,
    reasoning_effort="none",
)

openai_4o_mini_json = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-4o-mini",
    max_tokens=300,
    model_kwargs={"response_format": {"type": "json_object"}},
)

openai_gpt4o = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o",
)

# Primary tutoring model via xAI's OpenAI-compatible API. Requires XAI_API_KEY.
grok_4_5 = ChatOpenAI(
    model="grok-4.5",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

# DeepSeek V4 via langchain-deepseek. Requires DEEPSEEK_API_KEY.
_DEEPSEEK_THINKING = {"thinking": {"type": "enabled"}
, "reasoning_effort": "low"}

deepseek_v4_pro = ChatDeepSeek(
    model="deepseek-v4-pro",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    extra_body={"thinking": {"type": "disabled"}},
)

deepseek_v4_flash = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
    extra_body={"thinking": {"type": "disabled"}},
)

# Fallback tutoring model (Anthropic Sonnet 5).
# Sonnet 5 rejects temperature/top_p/top_k; omit sampling params.
# Disable adaptive thinking for lower latency/cost on fallback turns.
claude_sonnet_5 = ChatAnthropic(
    model="claude-sonnet-5",
    max_tokens=MAX_TOKENS,
    thinking={"type": "disabled"},
)

# Main tutoring LLM: Grok 4.5 primary, Sonnet 5 fallback
grok_with_sonnet_fallback = create_model_with_fallback(
    primary_model=grok_4_5,
    fallback_model=claude_sonnet_5,
)

# DeepSeek V4 Pro primary, grok fallback
deepseekv4_with_grok_fallback = create_model_with_fallback(
    primary_model=deepseek_v4_pro,
    fallback_model=grok_4_5,
)

deepseek_flash_with_fallback = create_model_with_fallback(
    primary_model=deepseek_v4_flash,
    fallback_model=openai_gpt56_luna,
)

# Back-compat alias used by app.py / chain wiring
claude_sonnet_with_fallback = grok_with_sonnet_fallback

claude_haiku = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

claude_haiku_with_fallback = create_model_with_fallback(
    primary_model=claude_haiku,
    fallback_model=openai_gpt56_luna,
)
