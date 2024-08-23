# In this file, all chains are defined with LC Expression Language 
# Doing so alone streaming of the outupt
# Created 2/21/2024
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser, json_output_parser

# On Aug 2024, the output_parser is moved to langchain_core.output_parsers.string
from langchain_core.output_parsers.string import StrOutputParser

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

output_parser = StrOutputParser()


# define the router chain
def router_chain(llm):
    query_router_template = """
    You are an AI query router for a coding course in business school. 
    The following is a user query: {query}. Based on the content of this query, determine its category according to the guidelines provided:

    - If the query is about the chat history, classify it as 0.
    - If the query requires specific knowledge, such as syllabus, assignments, lectures, classify it as 2.
    - For other queries including coding in Python, including syntax, libraries, and programming concepts, classify it as 1.

    Output the classification number without any additional text or explanation.
    """

    router_prompt = ChatPromptTemplate.from_template(query_router_template)
    setup = RunnableParallel(
        {"query": RunnablePassthrough()}
    )
    router_chain = setup | router_prompt | llm | output_parser

    return router_chain

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
    query_template = """
    You are a virtual teaching assistant name Dayton, for a graduate level Data and Decision Analytics course in the MBA core curriculum at Goizueta Business School. You facilitate the instructor with in-class activities that encourages analytical and critical thinking. Your task is to answer student query about data or decision analytics delimited by <query> tag. You should consider the chat history when relevant. Your response should be relevant and concise.
    
    Before generating a response, think step by step and adhere to the following guidelines:
    1. Determine the type of query: explanation, practice problems, or software implementation.
    2. Generate a response based on the query type:
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

    prompt = ChatPromptTemplate.from_template(query_template)

    setup = RunnableParallel(
        {"query": RunnablePassthrough(),
         "chat_history": RunnablePassthrough(),
         }
    )

    chain = setup | prompt | llm | output_parser

    return chain

# 3b. Setup LLMChain & prompts for RAG answer generation
def rag_chain(llm, retriever):
    template = """
    You are a virtual TA Dayton for MBA data analytics course in Goizueta Business School. Your task is to guide students step by step to complete student query delimited by <query> tag. You will generate the response ONLY based on retrieved context delimited by <context> tag. 
    
    Before generating a response, think step by step and adhere to the following guidelines:
    1. Read the retrieved context carefully and understand the content.
    2. Generate a response that best answer the query.
    
    Your response should be direct, concise and helpful, and adhere to the guidelines provided:
    - Answer the query directly
    - Say "I don't know" when the answer is not available in the context. 
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual.

    Query: <query>{query}</query>

    Retrieved context: <context>{context}</context>

    Your response:
    """
    # 
    # Please generate an appropriate response. Format the output when possible. 

    prompt = ChatPromptTemplate.from_template(template)
    setup_retrieval = RunnableParallel(
        {"context": retriever,
         "query": RunnablePassthrough(),
         }
    #     {
    #     "context": itemgetter("query") | retriever,
    #     "query": itemgetter("query"),
    #     "chat_history": itemgetter("chat_history"),
    # }
    )

    chain = setup_retrieval | prompt | llm | output_parser

    return chain

# 3c. Setup LLMChain & prompts for practice answer generation
def step_chain(llm, retriever):
    template = """
    You are a virtual TA Dayton for MBA data analytics course in Goizueta Business School. Your task is to guide students step by step to complete student query delimited by <query> tag. You will generate the response ONLY based on retrieved context delimited by <context> tag. 
    
    Before generating a response, think step by step and adhere to the following guidelines:
    1. Read the retrieved context carefully and understand the content.
    2. Develop a step by step plan to answer the query based on the context.
    3. Generate a response that guides the student to complete the task.
    
    You will follow the Socratic method. Your response should be concise and helpful, and adhere to the guidelines provided:
    - First provide the retrieved context as what has been discussed in class
    - generate a step-by-step to address the query
    - Help students make progress in the task
    - generate response in business context when possible,
    - Say "I don't know" when the answer is not available in the context. 
    - Limit response in 300 tokens or less.
    - Format the output when possible for better visual.

    Query: <query>{query}</query>

    Retrieved context: <context>{context}</context>

    Your response:
    """
    # 
    # Please generate an appropriate response. Format the output when possible. 

    prompt = ChatPromptTemplate.from_template(template)
    setup_retrieval = RunnableParallel(
        {"context": retriever,
         "query": RunnablePassthrough(),
         }
    )

    chain = setup_retrieval | prompt | llm | output_parser

    return chain

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
