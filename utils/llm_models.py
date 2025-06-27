from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from typing import Optional
from langchain_core.language_models import BaseChatModel
from pydantic import Field

# Create a couple of Global Variables
TEMPERATURE = 0.2
MAX_TOKENS = 512

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
        try:
            return self.primary.stream(*args, **kwargs)
        except Exception as e:
            self._log_fallback(e, "stream")
            return self.fallback.stream(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        try:
            return await self.primary.astream(*args, **kwargs)
        except Exception as e:
            self._log_fallback(e, "astream")
            return await self.fallback.astream(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return f"ModelWithFallback({self.primary._llm_type}->{self.fallback._llm_type})"

def create_model_with_fallback(
    primary_model: BaseChatModel,
    fallback_model: BaseChatModel
) -> BaseChatModel:
    """Creates a wrapper around the primary model that falls back to a secondary model if the primary fails"""
    return ModelWithFallback(primary=primary_model, fallback=fallback_model)

# Code generation llm with gpt-3.5
openai_gpt35 = ChatOpenAI(temperature=TEMPERATURE, 
                 model="gpt-3.5-turbo",
                 verbose=False,
                 max_tokens=300,
                 )

openai_gpt4o_mini = ChatOpenAI(temperature=TEMPERATURE, 
                 model="gpt-4o-mini",
                 verbose=False,
                 max_tokens=300,
                 )

openai_4o_mini_json = ChatOpenAI(temperature=TEMPERATURE,
        model="gpt-4o-mini",
        max_tokens=300,
        model_kwargs={ "response_format": { "type": "json_object" } }
        )

# Router llm: Choose OpenAI-GPT4 for better reasoning 
openai_gpt4 = ChatOpenAI(temperature=0.1, 
                 model="gpt-4-0125-preview",
                 verbose=False,
                 max_tokens=50,
                 )


# Create GPT-4 model for fallback
openai_gpt4o = ChatOpenAI(temperature=0.1, 
        model='gpt-4o',
        )

# Keep the original models for reference or direct use
claude_sonnet = ChatAnthropic(
        model='claude-3-5-sonnet-latest',
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )

# define the Anthropic chat client with fallback
claude_sonnet_with_fallback = create_model_with_fallback(
    primary_model=claude_sonnet,
    fallback_model=openai_gpt4o
)



claude_opus = ChatAnthropic(
        model='claude-3-opus-20240229',
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )


# Keep the original models for reference
claude_haiku = ChatAnthropic(
        model='claude-3-5-haiku-latest',
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
        )

# Create claude-haiku with fallback to gpt4o-mini
claude_haiku_with_fallback = create_model_with_fallback(
    primary_model=claude_haiku,
    fallback_model=openai_gpt4o_mini
)


