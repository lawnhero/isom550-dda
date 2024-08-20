

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain_openai import ChatOpenAI


# define the router chain
def router_chain(llm):
    query_router_template = """
    You are an AI router that decide how to generate response to user query. 
    If you're confident the query is about coding in Python, then you will return 1. 
    If you believe that the query requires additional or specific rather than generic knowledge, 
    then you will return 2. Your output should only include numbers. 

    Here is the user query: {query}
    """

    router_prompt = PromptTemplate(input_variables=["query"], template=query_router_template)

    router_chain = LLMChain(llm=llm, prompt=router_prompt)

    return router_chain

# define the openai chain
def openai_chain(llm):
    query_template = """
    You are an virtual teaching assistant for ISOM 352, Applied data anlytics with coding
    class at Goizueta Business School. Your task is to answer student query to your best capacity.
     
    Here is the query: {query}
    """

    router_prompt = PromptTemplate(input_variables=["query"], template=query_template)

    chain = LLMChain(llm=llm, prompt=router_prompt)

    return chain

# 3b. Setup LLMChain & prompts for RAG answer generation
def rag_chain(llm):
    template = """
    You are a teaching assistant for question-answering tasks. Answer following query using relevant context. 
    Query: {query}
    Context: {retrieved_info}
    Please generate an appropriate response. Format the output when possible. 
    """

    prompt = PromptTemplate(
        input_variables=["query", "retrieved_info"],
        template=template)

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain