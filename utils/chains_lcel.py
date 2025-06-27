# ISOM 550 DDA Virtual TA Chains
# All chains are defined with LangChain Expression Language (LCEL) for streaming support
# Created for MBA Data and Decision Analytics course

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from typing import Optional
from langchain.schema.language_model import BaseLanguageModel

# Global output parser
output_parser = StrOutputParser()


def _format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join([doc.page_content for doc in docs])


def _create_simple_chain(template: str, llm: BaseLanguageModel) -> any:
    """Create a simple chain with template and LLM."""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | output_parser


def _create_rag_chain(template: str, llm: BaseLanguageModel, retriever) -> any:
    """Create a RAG chain with retriever, template and LLM."""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel({
        "context": retriever | _format_docs,
        "query": RunnablePassthrough()
    })
    return setup | prompt | llm | output_parser


# =============================================================================
# MAIN CHAINS FOR MBA DATA ANALYTICS VIRTUAL TA
# =============================================================================

def class_chain(llm: BaseLanguageModel):
    """
    Chain for in-class activities and analytical thinking.
    Handles explanations, practice problems, and software implementation queries.
    """
    template = """You are Dayton, a virtual TA for MBA Data and Decision Analytics. You facilitate in-class activities that encourage analytical thinking.

**Your Role:**
- Answer queries about data/decision analytics only
- Provide business-focused responses for MBA students
- Consider chat history when relevant

**Response Guidelines:**
- **Explanations**: Provide clear, concise answers
- **Practice Problems**: Create 2 multiple-choice questions with code snippets, highlight correct answers with brief reasoning
- **Software Help**: Give direct implementation guidance for Excel, JMP, Python, or SQL

**Format**: Keep responses under 300 words, use clear formatting, exclude XML tags.

**Query**: {query}

**Chat History**: {chat_history}

**Response**:"""

    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel({
        "query": RunnablePassthrough(),
        "chat_history": itemgetter("chat_history")
    })
    return setup | prompt | llm | output_parser


def rag_chain(llm: BaseLanguageModel, retriever):
    """
    Chain for course logistics using RAG on course materials.
    Provides direct answers based on retrieved course content.
    """
    template = """You are Dayton, virtual TA for MBA Data Analytics at Goizueta Business School. Answer the student's query using ONLY the provided course materials.

**Instructions:**
1. Use only information from the retrieved context
2. Answer directly and concisely
3. Say "I don't have that information in the course materials" if not found

**Query**: {query}

**Course Materials**: {context}

**Answer**:"""

    return _create_rag_chain(template, llm, retriever)


def step_chain(llm: BaseLanguageModel, retriever):
    """
    Chain for assignment help using Socratic method.
    Guides students step-by-step without giving final answers.
    """
    template = """You are Dayton, a Socratic virtual TA for MBA Data Analytics. Guide students through their assignments using a step-by-step approach without giving final answers.

**Your Approach:**
1. Reference what's been covered in class (from context)
2. Provide ONLY the immediate next step
3. Encourage analytical thinking
4. Use business contexts when possible

**If topic not covered**: Say "I don't believe this topic is covered in our class materials"

**Query**: {query}

**Class Materials**: {context}

**Guidance**:
**What we've covered**: [Briefly reference relevant context]

**Your next step**: [Provide only the immediate next step, not the solution]"""

    return _create_rag_chain(template, llm, retriever)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_all_chains(claude_sonnet, claude_haiku, retriever_course, retriever_contents):
    """
    Convenience function to initialize all chains at once.
    
    Returns:
        dict: Dictionary containing all initialized chains
    """
    return {
        'class_chain': class_chain(claude_sonnet),
        'rag_chain': rag_chain(claude_haiku, retriever_course),
        'step_chain': step_chain(claude_sonnet, retriever_contents)
    }
