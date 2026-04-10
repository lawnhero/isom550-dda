from operator import itemgetter
from typing import Dict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field, validator

output_parser = StrOutputParser()


class Label(BaseModel):
    """Pydantic model for router output."""

    query: str = Field(description="Enhanced query for downstream chain")
    label: str = Field(description="Routing label")

    @validator("label")
    @classmethod
    def validate_label(cls, value):
        allowed_labels = ["course", "contents"]
        if value not in allowed_labels:
            raise ValueError(f"Label must be one of {allowed_labels}")
        return value


pydantic_parser = PydanticOutputParser(pydantic_object=Label)


def _format_docs(docs):
    """Format retrieved documents into a single string block."""
    return "\n\n".join([doc.page_content for doc in docs])


def format_chat_history(chat_history, max_messages: int = 8) -> str:
    """Convert chat history to readable text with a bounded window."""
    if not chat_history:
        return "No previous conversation."
    trimmed = chat_history[-max_messages:]
    lines = []
    for message in trimmed:
        role = "Student" if "Human" in str(type(message)) else "Assistant"
        lines.append(f"{role}: {message.content}")
    return "\n".join(lines)


def infer_learning_objective(query: str) -> str:
    """Infer a course objective tag from query keywords."""
    lowered = query.lower()
    objective_map = {
        "regression": "Regression modeling and interpretation",
        "hypothesis": "Hypothesis testing and statistical inference",
        "probability": "Probability foundations",
        "classification": "Predictive classification methods",
        "sql": "Data querying and transformation",
        "python": "Python analytics implementation",
        "excel": "Spreadsheet analytics workflows",
        "clustering": "Segmentation and clustering",
        "visual": "Data visualization for decision making",
        "grading": "Course policy and grading logistics",
        "deadline": "Course schedule and due date logistics",
    }
    for keyword, objective in objective_map.items():
        if keyword in lowered:
            return objective
    return "General data and decision analytics reasoning"


def infer_learner_level(chat_history) -> str:
    """Estimate learner level from interaction patterns."""
    if not chat_history:
        return "novice"
    student_messages = [
        msg.content.lower()
        for msg in chat_history
        if "Human" in str(type(msg))
    ]
    if not student_messages:
        return "novice"
    complexity_signals = sum(
        1
        for message in student_messages[-4:]
        if any(token in message for token in ["assumption", "coefficient", "p-value", "multicollinearity", "sql join"])
    )
    if complexity_signals >= 2:
        return "advanced"
    if complexity_signals == 1:
        return "intermediate"
    return "novice"


def detect_attempt_check(query: str) -> bool:
    """Detect if student is asking for attempt-level feedback."""
    patterns = ["check my", "is this right", "my answer", "my attempt", "i tried", "did i do"]
    lowered = query.lower()
    return any(pattern in lowered for pattern in patterns)


def build_learning_profile(query: str, response_mode: str, chat_history) -> Dict[str, str]:
    """Build tutoring profile for adaptive response behavior."""
    return {
        "response_mode": response_mode or "Teach me step-by-step",
        "learning_objective": infer_learning_objective(query),
        "learner_level": infer_learner_level(chat_history),
        "attempt_check": "yes" if detect_attempt_check(query) else "no",
    }


def _create_simple_chain(template: str, llm: BaseLanguageModel, parser=output_parser):
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | parser


def create_routing_chain(llm: BaseLanguageModel):
    """Create a routing chain that separates logistics and learning help."""
    template = """You are a query router for a data analytics Virtual TA.

Current Query: <query>{query}</query>
Previous Conversation: {chat_history}

Classify into one label:
- course: deadlines, grading, schedule, policy, syllabus logistics
- contents: analytics concepts, assignment help, coding, interpretation

Rules:
- Preserve user meaning, rewrite the query for clarity and specificity.
- Prefer contents if unsure.
- Return strict JSON only.

Response format:
{{
  "query": "rewritten query",
  "label": "course or contents"
}}"""
    return _create_simple_chain(template, llm, parser=pydantic_parser)


def call_function(
    tool_name: str,
    query: str,
    chains_dict: dict,
    chat_history=None,
    memory_summary: str = "",
    response_mode: str = "Teach me step-by-step",
):
    """Invoke the appropriate chain based on router label."""
    history_text = format_chat_history(chat_history, max_messages=8)
    profile = build_learning_profile(query, response_mode, chat_history)
    payload = {
        "query": query,
        "chat_history": history_text,
        "memory_summary": memory_summary or "No memory summary yet.",
        "response_mode": profile["response_mode"],
        "learning_objective": profile["learning_objective"],
        "learner_level": profile["learner_level"],
        "attempt_check": profile["attempt_check"],
    }

    if "course" in tool_name:
        return chains_dict["rag_chain"].stream(payload)
    if "contents" in tool_name:
        return chains_dict["step_chain"].stream(payload)
    return chains_dict["class_chain"].stream(payload)


def unified_ta_chain_with_tools(
    main_llm: BaseLanguageModel,
    router_llm: BaseLanguageModel,
    retriever_course,
    retriever_contents,
):
    chain_dict = {
        "rag_chain": rag_chain(main_llm, retriever_course),
        "step_chain": step_chain(main_llm, retriever_contents),
        "class_chain": class_chain(main_llm),
    }
    router = create_routing_chain(router_llm)
    return router, chain_dict


def class_chain(llm: BaseLanguageModel):
    template = """You are Dayton, a Virtual TA for MBA Data and Decision Analytics.

Learning objective: {learning_objective}
Estimated learner level: {learner_level}
Preferred response mode: {response_mode}
Attempt check requested: {attempt_check}

Response contract (strict):
1) Keep focus on analytics and business decision-making.
2) Do not mention internal settings like response_mode, learner_level, or learning objective.
3) Match response style exactly:
   - Direct answer: use this exact structure:
     **Answer**
     <concise answer in <=120 words>
     **Check yourself**
     - <one short verification action>
   - Hint-first: use this exact structure:
     **Hints**
     - Hint 1: <hint>
     - Hint 2: <optional hint>
     **Your turn**
     - <one action student should do next>
     (No final answer unless student explicitly asks again.)
   - Teach me step-by-step: use this exact structure:
     **Step 1**
     - Do: <single actionable step>
     - Why: <short reason>
     **Checkpoint**
     - <what student should observe or produce>
4) If attempt_check is yes, give rubric feedback with:
   - What is correct
   - What to fix
   - One next action
5) Keep total response <=180 words, with short bullets when useful.
6) End with one brief follow-up question that moves learning forward.

Conversation memory summary:
{memory_summary}

Recent chat:
{chat_history}

Student query:
{query}

Response:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
            "memory_summary": itemgetter("memory_summary"),
            "response_mode": itemgetter("response_mode"),
            "learning_objective": itemgetter("learning_objective"),
            "learner_level": itemgetter("learner_level"),
            "attempt_check": itemgetter("attempt_check"),
        }
    )
    return setup | prompt | llm | output_parser


def rag_chain(llm: BaseLanguageModel, retriever):
    template = """You are Dayton, a Virtual TA for BUS 350 Data and Decision Analytics.
Answer the logistics question using ONLY the provided course materials.

Learning objective: {learning_objective}
Preferred response mode: {response_mode}

Rules:
1) Use only retrieved course context; if insufficient, say you do not have enough information.
2) Do not invent policies, deadlines, grading rules, or logistics details.
3) Keep response <=120 words and plain language.
4) Style output templates:
   - Direct answer:
     **Answer**
     <direct logistics answer>
     **Check yourself**
     - <one place to verify>
   - Hint-first:
     **Hints**
     - <where to look in course materials>
     - <what keyword to search>
     **Your turn**
     - <one verification action>
   - Teach me step-by-step:
     **Step 1**
     - <first verification step>
     **Checkpoint**
     - <what to confirm before next step>
6) Do not mention internal settings or hidden context fields.

Conversation memory summary:
{memory_summary}

Recent chat:
{chat_history}

Retrieved context:
{context}

Query:
{query}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "context": itemgetter("query") | retriever | _format_docs,
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
            "memory_summary": itemgetter("memory_summary"),
            "response_mode": itemgetter("response_mode"),
            "learning_objective": itemgetter("learning_objective"),
        }
    )
    return setup | prompt | llm | output_parser


def step_chain(llm: BaseLanguageModel, retriever):
    template = """You are Dayton, a Socratic Virtual TA for BUS 350 Data and Decision Analytics.

Learning objective: {learning_objective}
Estimated learner level: {learner_level}
Preferred response mode: {response_mode}
Attempt check requested: {attempt_check}

Guidance policy (strict):
1) Move the learner forward by exactly one meaningful step.
2) Use class context and prior progress; do not repeat prior completed steps.
3) Style behavior:
   - Direct answer: use this format:
     **Answer**
     <concise answer in <=140 words>
     **Verify**
     - <one action to validate understanding>
   - Hint-first: use this format:
     **Hints**
     - Hint 1: <hint>
     - Hint 2: <optional hint>
     **Your turn**
     - <small next action>
     (No final numeric/code result.)
   - Teach me step-by-step: use this format:
     **Step 1**
     - Do: <single action>
     - Why: <brief reason>
     **Expected output**
     - <what student should get>
4) If attempt_check is yes, provide rubric feedback:
   - Correct parts
   - Incorrect/missing parts
   - One revision to try next
5) If the student asks for full solution, refuse politely and provide the next actionable hint.
6) If topic is out of scope, say it is not covered in class materials and suggest the nearest covered topic.
7) Do not mention internal settings (response_mode, learner_level, objective tags).
8) End with one short question that confirms readiness for the next step.

Conversation memory summary:
{memory_summary}

Recent chat:
{chat_history}

Class materials:
{context}

Student query:
{query}

Response:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "context": itemgetter("query") | retriever | _format_docs,
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
            "memory_summary": itemgetter("memory_summary"),
            "response_mode": itemgetter("response_mode"),
            "learning_objective": itemgetter("learning_objective"),
            "learner_level": itemgetter("learner_level"),
            "attempt_check": itemgetter("attempt_check"),
        }
    )
    return setup | prompt | llm | output_parser


def recap_chain(llm: BaseLanguageModel):
    template = """You are a learning recap assistant.
Summarize the current session for the student in this exact structure:
- What you now know
- What to try next
- Common pitfalls

Keep it under 120 words.

Conversation memory summary:
{memory_summary}

Recent chat:
{chat_history}

Recap:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | output_parser


def summarize_memory_chain(llm: BaseLanguageModel):
    template = """Create a compact learning memory summary for continuity.
Previous summary:
{previous_summary}

Recent turns:
{recent_turns}

Return 4 bullets max, focused on:
- learner goal
- progress made
- misconceptions/open issues
- best next step
"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | output_parser


def get_all_chains(
    claude_sonnet,
    claude_haiku,
    retriever_course,
    retriever_contents,
    router_llm=None,
):
    if router_llm is None:
        router_llm = claude_sonnet

    router, chain_dict = unified_ta_chain_with_tools(
        main_llm=claude_sonnet,
        router_llm=router_llm,
        retriever_course=retriever_course,
        retriever_contents=retriever_contents,
    )

    return {
        "class_chain": class_chain(claude_sonnet),
        "rag_chain": rag_chain(claude_haiku, retriever_course),
        "step_chain": step_chain(claude_sonnet, retriever_contents),
        "recap_chain": recap_chain(claude_haiku),
        "summary_chain": summarize_memory_chain(claude_haiku),
        "router": router,
        "chain_dict": chain_dict,
    }
