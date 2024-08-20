# add the temperature and max_tokens parameters to the models 5/10/2024
# Allows more flexibility in the models

from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

# define a class of chat models
class LLMModels():
    def __init__(self, temperature=0.5, max_tokens=1000):
        self.temperature = temperature
        self.max_tokens = max_tokens


    # define openai gpt3.5 turbo
    def openai_gpt35(self,temperature=0.5):
        openai_gpt35 = ChatOpenAI(
            model='gpt-3.5-turbo',
            temperature=temperature,
            # max_tokens=self.max_tokens,,
            )
        return openai_gpt35
    
    # define openai gpt3.5 turbo
    def openai_gpt4o(self,temperature=0.5):
        openai_gpt4o = ChatOpenAI(
            model='gpt-4o',
            temperature=temperature,
            # max_tokens=self.max_tokens,,
            )
        return openai_gpt4o
    
    # define openai gpt4 turbo
    def openai_gpt4(self, temperature=0.5):
        openai_gpt4 = ChatOpenAI(
            model='gpt-4-turbo',
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return openai_gpt4
    
    # define claude sonnet
    def claude_sonnet(self, temperature=0.5):
        claude_sonnet = ChatAnthropic(
            model='claude-3-sonnet-20240229',
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return claude_sonnet
    
    # define claude sonnet
    def claude_sonnet35(self, temperature=0.5):
        claude_sonnet = ChatAnthropic(
            model='claude-3-5-sonnet-20240620',
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return claude_sonnet
    
    # define claude haiku
    def claude_haiku(self, temperature=0.5):
        claude_haiku = ChatAnthropic(
            model='claude-3-haiku-20240307',
            temperature=self.temperature,
            # max_tokens=self.max_tokens,
            )
        return claude_haiku
    
    # define claude opus
    def claude_opus(self, temperature=0.5):
        claude_opus = ChatAnthropic(
            model='claude-3-opus-20240229',
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return claude_opus
    
    # define groq mixtral
    def groq_mixtral(self, temperature=0.5):
        model = ChatGroq(
            model_name='mixtral-8x7b-32768',
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return model
    
    # define groq llama3
    def groq_llama3(self, temperature=0.5):
        model = ChatGroq(
            model_name='llama3-70b-8192',
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return model
    
    def llama3(self, temperature=0.5):
        model = ChatOllama(
            model='llama3:latest',
            base_url="http://localhost:11434",
            temperature=temperature,
            # max_tokens=self.max_tokens,
            )
        return model
 
    