# In this file, all chains are defined with LC Expression Language 
# Doing so alone streaming of the outupt
# Created 2/21/2024
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser, json_output_parser

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

from typing import Dict, Any, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema.language_model import BaseLanguageModel

output_parser = StrOutputParser()

json_parser = JsonOutputParser()


def _create_chain(template: str, output_parser: Any, llm: Optional[BaseLanguageModel] = None) -> Any:
    """Create a generic chain with the given template, output parser, and LLM."""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {key: RunnablePassthrough() for key in prompt.input_variables}
    )
    chain_llm = llm         
    return setup | prompt | chain_llm | output_parser

    # return _create_chain(router_template, json_parser, llm)

def _format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])

# define the router chain
def router_chain(llm):
    template = """
    You are an AI query router for a coding course in business school. 
    The following is a user query: {query}. Based on the content of this query, determine its category according to the guidelines provided:

    - If the query is about the chat history, classify it as 0.
    - If the query requires specific knowledge, such as syllabus, assignments, lectures, classify it as 2.
    - For other queries including coding in Python, including syntax, libraries, and programming concepts, classify it as 1.

    Output the classification number without any additional text or explanation.
    """

    return _create_chain(template, output_parser, llm)

def query_analysis_chain(llm):
    template = """
    You are an expert AI assistant who specialize in rewriting user query in the context of an introductory Python coding class in a top Business School. Your task is to analyze the user query and determine its category based on the guidelines provided."""

    prompt = ChatPromptTemplate.from_template(template)

    setup = RunnableParallel(
        {"query": RunnablePassthrough(),
         }
    )

    chain = setup | prompt | llm | output_parser

# define the openai chain
def class_chain(llm):
    template = """
    You are a virtual teaching assistant name Dayton, for a MBA Data and Decision Analytics course. You facilitate the instructor with in-class activities that encourages analytical and critical thinking. Your task is to answer student query about data or decision analytics delimited by <query> tag. You should consider the chat history when relevant. Your response should be relevant and concise.
    
    before generating a response adhere to the following guidelines:
    1. Understand the query and the context of the chat history.
    2. Determine the type of query: explanation, practice problems, or software implementation.
    3. Generate a response based on the query type:
        - if the query is about clarification or explanation, answer the query to your best ability. 
        - If the query asks for practice problems or exercises, generate no more than two questions in multiple choice format with one correct answer. Include code snippets for each question when possible. Highlight the correct answer and provide a brief reasoning. 
        - If the query asks for software implementation in Excel or JMP, or SQL, provide a direct answer.

    Your response should be concise and helpful to MBA students with no background in analytics, and adhere to the guidelines provided:
    - ONLY answer queries related to data or decision analytics,
    - Only provide direct answers to the query based on the query type
    - generate response in business context when possible,
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual.
    - Exclude any xml tags.

    Student query: <query>{query} </query>

    Consider the chat history: {chat_history}.

    Generate your response:

    """

    return _create_chain(template, output_parser, llm)

# 3b. Setup LLMChain & prompts for RAG answer generation
def rag_chain(llm, retriever):
    template = """
    You are a virtual TA Dayton for MBA data analytics course in Goizueta Business School. Your task is to guide students step by step to complete student query delimited by <query> tag. You will generate the response ONLY based on retrieved context delimited by <context> tag. 
    
    Before generating a response, think step by step and adhere to the following guidelines:
    1. Read the retrieved context carefully and understand the content.
    2. Evaluate the query in the context.
    3. Generate a response that best answer the query.
    
    Your response should be direct, concise and helpful, and adhere to the guidelines provided:
    - Answer the query directly
    - Say "I don't know" when the answer is not available in the context. 
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual.

    Query: <query>{query}</query>

    Retrieved context: <context>{context}</context>

    Your response:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    setup = RunnableParallel(
        {
            "context": retriever | _format_docs,
            "query": RunnablePassthrough()
        }
    )
    
    return setup | prompt | llm | output_parser

# 3c. Setup LLMChain & prompts for practice answer generation
def step_chain(llm, retriever):
    template = """
    You are a Socratic virtual TA for MBA data analytics course in Goizueta Business School. You take a Socratic approach to facilitate analytical thinking. Your task is to guide students step by step to complete student query delimited by <query> tag. You will generate the response ONLY based on retrieved context delimited by <context> tag. 
    
    Before generating a response, think step by step and adhere to the following guidelines:
    1. Read the retrieved context carefully and understand the content.
    2. Develop a step by step plan to complete the query based on the context.
    3. Generate a response that guides the student with the immediate next step.
    
    Your response should be concise and helpful, and adhere to the guidelines provided:
    - First provide the retrieved context as what has been discussed in class
    - generate ONLY the immediate next step for completing the query, but do not provide the final answer.
    - generate response in business context when possible,
    - Say "I don't believe the topic is covered in class" when the answer is not available in the context. 
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual.

    Query: <query>{query}</query>

    Retrieved context: <context>{context}</context>

    Your response:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    setup = RunnableParallel(
        {
            "context": retriever | _format_docs,
            "query": RunnablePassthrough()
        }
    )
    
    return setup | prompt | llm | output_parser

# define chat history chain
# 3d. Setup LLMChain & prompts for RAG answer generation
def chat_history_chain(llm):
    template = """
    You're my AI assistant that answer queries based on chat hisotry. 
    Your response should be direct, concise and helpful.
    Answer the user query: {query} 
    Here is the chat history: {chat_history}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = prompt | llm | output_parser

    return chain
