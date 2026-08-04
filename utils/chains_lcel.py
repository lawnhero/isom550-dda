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


CURRICULUM_TOPICS = [
    "Descriptive statistics",
    "Probability",
    "Hypothesis testing",
    "Regression",
    "Decision analysis",
    "SAS JMP / Excel workflows",
]

# Subtopics shown after a student picks a top-level curriculum topic.
CURRICULUM_SUBTOPICS = {
    "Descriptive statistics": [
        "Mean / median / mode",
        "Variance and standard deviation",
        "Distributions and shape",
        "Percentiles and boxplots",
        "Exploratory data analysis",
    ],
    "Probability": [
        "Basic probability rules",
        "Conditional probability",
        "Bayes theorem",
        "Random variables",
        "Expected value",
    ],
    "Hypothesis testing": [
        "Null vs alternative hypotheses",
        "p-values and significance",
        "t-tests",
        "ANOVA",
        "Type I / Type II errors",
    ],
    "Regression": [
        "Simple linear regression",
        "Multiple regression",
        "Coefficient interpretation",
        "R-squared and fit",
        "Assumptions and diagnostics",
        "Multicollinearity",
    ],
    "Decision analysis": [
        "Decision trees",
        "Expected value of decisions",
        "Sensitivity analysis",
        "Payoff tables",
        "Value of information",
    ],
    "SAS JMP / Excel workflows": [
        "JMP basics",
        "Excel Data Analysis ToolPak",
        "Building a model in Excel",
        "Reading JMP / Excel output",
        "Common workflow tips",
    ],
}

# Backward-compatible alias used by topic pills in the UI.
COURSE_TOPIC_CHOICES = CURRICULUM_TOPICS

_TOPIC_KEYWORDS = {
    "Descriptive statistics": [
        "descriptive",
        "distribution",
        "mean",
        "median",
        "variance",
        "standard deviation",
    ],
    "Probability": ["probability", "bayes", "conditional", "random variable"],
    "Hypothesis testing": ["hypothesis", "p-value", "significance", "t-test", "anova"],
    "Regression": ["regression", "coefficient", "r-squared", "multicollinearity"],
    "Decision analysis": [
        "excel model",
        "sensitivity analysis",
        "decision tree development / solution",
    ],
    "SAS JMP / Excel workflows": ["jmp", "excel", "data analysis toolpak"],
}


def get_subtopics(topic: str) -> list:
    """Return subtopic labels for a curriculum topic, or an empty list."""
    return list(CURRICULUM_SUBTOPICS.get((topic or "").strip(), []))


def format_topic_focus(topic: str = "", subtopic: str = "") -> str:
    """Build a display/query focus string from topic and optional subtopic."""
    topic = (topic or "").strip()
    subtopic = (subtopic or "").strip()
    if topic and subtopic:
        return f"{topic}: {subtopic}"
    return subtopic or topic

_LOGISTICS_KEYWORDS = {
    "grading": "Course policy and grading logistics",
    "deadline": "Course schedule and due date logistics",
    "syllabus": "Course schedule and due date logistics",
    "assignment due": "Course schedule and due date logistics",
}


def infer_curriculum_topic(query: str) -> str:
    """Return the best-matching curriculum topic label, or empty string."""
    lowered = (query or "").lower()
    for topic in CURRICULUM_TOPICS:
        if topic.lower() in lowered:
            return topic
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return topic
    return ""


def infer_learning_objective(query: str) -> str:
    """Infer a course objective tag from query keywords."""
    topic = infer_curriculum_topic(query)
    if topic:
        return topic
    lowered = query.lower()
    for keyword, objective in _LOGISTICS_KEYWORDS.items():
        if keyword in lowered:
            return objective
    return "General data and decision analytics reasoning"


def infer_topic_from_history(chat_history, max_messages: int = 6) -> str:
    """Return a concrete topic from recent chat, or empty string if unclear."""
    if not chat_history:
        return ""
    recent = chat_history[-max_messages:]
    for message in reversed(recent):
        if "Human" not in str(type(message)):
            continue
        topic = infer_curriculum_topic(message.content)
        if topic:
            return topic
        objective = infer_learning_objective(message.content)
        if objective in CURRICULUM_TOPICS:
            return objective
    for message in reversed(recent):
        if "AI" not in str(type(message)) and "Assistant" not in str(type(message)):
            continue
        topic = infer_curriculum_topic(message.content)
        if topic:
            return topic
        objective = infer_learning_objective(message.content)
        if objective in CURRICULUM_TOPICS:
            return objective
    return ""


def compose_quick_action_query(intent: str, topic: str = "", attempt_text: str = "") -> str:
    """Build a concrete tutoring request from a quick-action intent."""
    topic = (topic or "").strip()
    attempt_text = (attempt_text or "").strip()
    if intent == "explain":
        return (
            f"Explain the {topic} topic clearly with a simple business example."
        )
    if intent == "practice":
        return (
            f"Create one practice question on the {topic} topic, then guide me with hints. "
            "Stay strictly on this topic; do not invent an unrelated scenario."
        )
    if intent == "check":
        if attempt_text:
            return (
                "Please check my attempt and tell me what to fix next.\n\n"
                f"My attempt:\n{attempt_text}"
            )
        return "Please check my attached attempt and tell me what to fix next."
    if intent == "next_step":
        return "Based on our conversation so far, what is my immediate next learning step?"
    return topic or attempt_text



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


def build_chain_payload(
    query: str,
    chat_history=None,
    response_mode: str = "Teach me step-by-step",
    context: str = "",
) -> Dict[str, str]:
    """Build the common payload passed into tutoring chains."""
    history_text = format_chat_history(chat_history, max_messages=8)
    profile = build_learning_profile(query, response_mode, chat_history)
    return {
        "query": query,
        "chat_history": history_text,
        "response_mode": profile["response_mode"],
        "learning_objective": profile["learning_objective"],
        "learner_level": profile["learner_level"],
        "attempt_check": profile["attempt_check"],
        "context": context or "No retrieved context available.",
    }


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
            "response_mode": itemgetter("response_mode"),
            "learning_objective": itemgetter("learning_objective"),
            "learner_level": itemgetter("learner_level"),
            "attempt_check": itemgetter("attempt_check"),
        }
    )
    return setup | prompt | llm | output_parser


def rag_chain(llm: BaseLanguageModel):
    template = """You are Dayton, a Virtual TA for Data and Decision Analytics.
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
            "context": itemgetter("context"),
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
            "response_mode": itemgetter("response_mode"),
            "learning_objective": itemgetter("learning_objective"),
        }
    )
    return setup | prompt | llm | output_parser


def facts_chain(llm: BaseLanguageModel):
    """Answer course-fact questions from the Tier A context block.

    No retrieval: the context block IS the source. Deliberately ignores
    response_mode -- a student asking when the final is due wants the date, not
    a hint or a guided exercise, whatever tutoring style they picked.
    """
    template = """You are Dayton, the Virtual TA for ISOM 550 Data and Decision Analytics.

Answer using ONLY the COURSE CONTEXT below. It is the authoritative record for
dates, people, grading, materials, and what has been covered in class so far.

Rules:
1) Use only the COURSE CONTEXT. If the answer is not there, say you do not have
   that information and point the student to Canvas, the syllabus, or the
   instructor. Never guess a deadline, policy, office hour, or grade weight.
2) Dates and times in the context are already in course-local time. Repeat them
   exactly as written. Do not convert, recompute, or infer any date, and do not
   work out what "this week" means beyond what the context states.
3) If the context begins with a "!! SCHEDULE RELIABILITY" block, follow the
   instruction on its "->" line before answering.
4) Link to Canvas whenever the context provides a URL, so the student can confirm.
5) Under 120 words, plain language.
6) Never mention internal settings, hidden fields, or that you were given a context block.

Always use this shape, regardless of the student's tutoring style preference:

**Answer**
<the fact, stated plainly>
**Confirm**
- <where to verify, with the Canvas link when the context has one>

COURSE CONTEXT:
{course_context}

Recent chat:
{chat_history}

Question:
{query}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "course_context": itemgetter("course_context"),
            "chat_history": itemgetter("chat_history"),
            "query": itemgetter("query"),
        }
    )
    return setup | prompt | llm | output_parser


def software_chain(llm: BaseLanguageModel):
    """Answer JMP / Excel how-to questions from the model's own knowledge.

    No retrieval by design: the model knows these tools better than any course
    index could teach it, and feeding it 4 loosely-matching stats Q&A rows as
    "context" actively misleads it -- which is what happened to ~92 logged JMP
    questions before this route existed.

    Ignores response_mode. This course teaches analytics, not JMP; withholding
    a menu path behind a hint wastes the student's time without teaching
    anything the course is actually assessing.
    """
    template = """You are Dayton, the Virtual TA for ISOM 550 Data and Decision Analytics.

The student needs help operating software. Answer from your own knowledge of the
tool, grounded by the course details below.

Rules:
1) Give concrete, numbered steps naming the exact menus, dialogs, and buttons.
2) State which version you are assuming, and add one short line noting menus may
   differ in other versions.
3) NEVER invent a menu path. If you are not confident about the exact location of
   a command in this version, say which part you are unsure of and point the
   student to the course walkthrough or the TA. A confident wrong click path
   costs more time than an honest "I'm not certain where this sits in 18".
4) If COURSE CONVENTIONS below contradict the tool's default behaviour, follow
   the course convention and say so explicitly -- this is where students most
   often misread their own output.
5) If a COURSE WALKTHROUGH matches the task, link it; the instructor's own
   version is better than a generic one.
6) If the student asks about software this course does not use, say which tool
   the course uses for that task instead of answering for the other tool.
7) Stay pointed at the analytics goal. Explain what the output means, briefly,
   not just where to click.
8) Never mention internal settings or hidden context fields.

{software_context}

Recent chat:
{chat_history}

Question:
{query}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "software_context": itemgetter("software_context"),
            "chat_history": itemgetter("chat_history"),
            "query": itemgetter("query"),
        }
    )
    return setup | prompt | llm | output_parser


def doc_chain(llm: BaseLanguageModel):
    """Answer from Tier C course documents: class recaps and assignment briefs.

    Unlike facts_chain, this one honours response_mode -- but only for the
    explanation it adds. What the document actually says is always reported
    plainly, because a student asking what an assignment requires needs the
    requirements, not a hint.
    """
    template = """You are Dayton, the Virtual TA for ISOM 550 Data and Decision Analytics.

Answer using the COURSE DOCUMENTS below. They are class recap announcements and
assignment instructions written by the instructor.

Rules:
1) Report what the documents say. Never invent a task, deliverable, point value,
   file name, or claim about what a class covered.
2) If the documents do not cover the question, say so plainly and suggest where
   to look. Do not fill the gap from general knowledge.
3) Name the document you are drawing on ("Class 9 (7/27) Sensitivity Analysis")
   and include its link when one is provided.
4) State the document's content plainly first. Then adapt any FURTHER
   explanation to the response mode:
   - Direct answer: add a one-line summary of what matters most.
   - Hint-first: after stating the requirements, ask one question that helps the
     student decide their next step.
   - Teach me step-by-step: after stating the requirements, break them into an
     ordered plan of what to do first, second, third.
5) Under 200 words unless the student asked for a full task list, in which case
   list every task.
6) Never mention internal settings, hidden fields, or that you were given documents.

Preferred response mode: {response_mode}

COURSE DOCUMENTS:
{context}

Recent chat:
{chat_history}

Question:
{query}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "context": itemgetter("context"),
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
            "response_mode": itemgetter("response_mode"),
        }
    )
    return setup | prompt | llm | output_parser


def step_chain(llm: BaseLanguageModel):
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
            "context": itemgetter("context"),
            "query": itemgetter("query"),
            "chat_history": itemgetter("chat_history"),
            "response_mode": itemgetter("response_mode"),
            "learning_objective": itemgetter("learning_objective"),
            "learner_level": itemgetter("learner_level"),
            "attempt_check": itemgetter("attempt_check"),
        }
    )
    return setup | prompt | llm | output_parser


def practice_chain(llm: BaseLanguageModel):
    template = """You are Dayton, a Virtual TA for MBA Data and Decision Analytics.

Topic: {topic}
Difficulty: {difficulty}
Learning objective: {learning_objective}
Estimated learner level: {learner_level}

Create exactly ONE practice question for this topic.
Rules:
1) Use a realistic MBA/business scenario tied to the topic.
2) Ask one clear question the student can answer in 3-5 sentences or a short calculation.
3) Do NOT provide the full solution.
4) End with one short hint the student can use if stuck.
5) Keep total response <=180 words.

Use this structure:
**Practice question**
<scenario + question>

**Hint**
- <one actionable hint>

Recent chat:
{chat_history}

Response:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "topic": itemgetter("topic"),
            "difficulty": itemgetter("difficulty"),
            "learning_objective": itemgetter("learning_objective"),
            "learner_level": itemgetter("learner_level"),
            "chat_history": itemgetter("chat_history"),
        }
    )
    return setup | prompt | llm | output_parser


def check_chain(llm: BaseLanguageModel):
    template = """You are Dayton, a Virtual TA for MBA Data and Decision Analytics.

Topic: {topic}
Learning objective: {learning_objective}
Estimated learner level: {learner_level}

Check the student's attempt and give constructive feedback.
Rules:
1) Use this exact structure:
   **What is correct**
   - <bullet(s)>
   **What to fix**
   - <bullet(s)>
   **Next action**
   - <one concrete revision step>
2) Be specific and encouraging; do not rewrite the full solution unless the attempt is blank.
3) Keep total response <=180 words.
4) End with one short follow-up question.

Recent chat:
{chat_history}

Student attempt:
{attempt_text}

Response:"""
    prompt = ChatPromptTemplate.from_template(template)
    setup = RunnableParallel(
        {
            "topic": itemgetter("topic"),
            "attempt_text": itemgetter("attempt_text"),
            "learning_objective": itemgetter("learning_objective"),
            "learner_level": itemgetter("learner_level"),
            "chat_history": itemgetter("chat_history"),
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

Recent chat:
{chat_history}

Recap:"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | output_parser


def get_all_chains(main_llm, light_llm):
    return {
        "class_chain": class_chain(main_llm),
        "rag_chain": rag_chain(light_llm),
        # Reading a fact out of a context block and linking Canvas is a light
        # task; the tutoring model is not needed for it.
        "facts_chain": facts_chain(light_llm),
        # Reporting what an assignment brief or class recap says, then adapting
        # the follow-on explanation, is real tutoring work -- main model.
        "doc_chain": doc_chain(main_llm),
        # Procedural steps grounded by a short context block -- light task, and
        # this is high-traffic (92 JMP questions in the logged history).
        "software_chain": software_chain(light_llm),
        "step_chain": step_chain(main_llm),
        "practice_chain": practice_chain(main_llm),
        "check_chain": check_chain(main_llm),
        "recap_chain": recap_chain(light_llm),
    }
