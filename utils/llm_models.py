from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
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

# Two providers, two keys: OPENAI_API_KEY and DEEPSEEK_API_KEY.
#
# GPT Luna does the routing (tool calling) and is the fallback for both
# DeepSeek tutors. DeepSeek V4 writes the answers: Pro for the tutoring chains,
# Flash for the light, high-traffic routes (facts, software steps).

# Dispatch build: tool calling, with a budget sized for tool-call JSON.
openai_gpt56_luna = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-5.6-luna",
    max_tokens=ROUTER_MAX_TOKENS,
    reasoning_effort="none",  # required for /v1/chat/completions + tool calling
)

# Full-budget build of the same model. Every screenshot turn is pinned here
# (transcribing a regression table back to the student needs the room), and
# it is what both DeepSeek wrappers fall back to.
openai_gpt56_luna_full = ChatOpenAI(
    temperature=TEMPERATURE,
    model="gpt-5.6-luna",
    max_tokens=MAX_TOKENS,
    reasoning_effort="none",
)

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

# Main tutoring model: concept, documents, practice, coach, check.
deepseek_pro_with_fallback = create_model_with_fallback(
    primary_model=deepseek_v4_pro,
    fallback_model=openai_gpt56_luna_full,
)

# Light routes: facts and software steps. Wrapped so a DeepSeek outage
# degrades to a slower answer rather than to no deadline answer.
deepseek_flash_with_fallback = create_model_with_fallback(
    primary_model=deepseek_v4_flash,
    fallback_model=openai_gpt56_luna_full,
)
