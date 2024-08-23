import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage

from langchain.globals import set_verbose
import chains_lcel as chains
from sidebar import sidebar
from llm_models import LLMModels

# Enable verbose logging
set_verbose(True)

# Set the page_title
st.set_page_config(
        page_title="ISOM 550 DDA Virtual TA - Beta", page_icon="🔍", layout="wide")

# cache the vectorized embedding database 
@st.cache_resource
# load the vectorized database
def load_db(db_path, embedding_model='text-embedding-ada-002'):
    embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=1)
    db_loaded = FAISS.load_local(db_path, embeddings,
                                 allow_dangerous_deserialization=True
                                 )
    
    return db_loaded

# 1. Load the Vectorised database
kb_db_path = 'data/emb_db'
db = load_db(kb_db_path)

# 2. Function for similarity search
retriever = db.as_retriever()

# 3. Setup LLM and chains
# initialize the llm
haiku = LLMModels().claude_haiku(temperature=0)
sonnet = LLMModels().claude_sonnet(temperature=0)
sonnet35 = LLMModels().claude_sonnet35(temperature=0)
gpt4o = LLMModels().openai_gpt4o(temperature=0)
# llm = LLMModels().claude_opus(temperature=0)
# llm = LLMModels().openai_gpt35(temperature=0)

# 3 Setup the various chains to perform various functions
step_chain = chains.step_chain(sonnet35, retriever)
# 3b. Setup LLMChain & prompts for RAG answer generation
rag_chain = chains.rag_chain(sonnet35, retriever)

# 3c. Setup direct openai_chain
# chat_chain = chains.class_chain(llm_gpt35)
chat_chain = chains.class_chain(sonnet35)
        
# 5. Build an app with streamlit
def main():

    st.header("🦜 Virtual TA - ISOM 550 DDA")
    # st.write("Currently support queries on syllabus and coding request.")
    sidebar()
    
    # Set up the radio button toggle with descriptions on both sides
    option = st.radio(
        label="Choose an option:",
        options=["In-Class", "Assignment", "Course Logistics"],  # Replace with your actual options
        index=0,  # Default selected option
        horizontal=True  # Display options horizontally
    ).lower()
    # Display the selected option
    # st.write(f"You selected: {option}")

    if "in-class" in option:
        initial_text = "Hello there. How can I help you with the in-class activity today? "
    else:
        initial_text = "Hello there. How can I help you with the assignment today? "

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(initial_text)]

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🦜"):
                st.markdown(message.content)
    
    # truncate chat history to last 5 messages
    max_num_messages = 2
    if len(st.session_state.chat_history) > max_num_messages:
        st.session_state.chat_history = st.session_state.chat_history[-max_num_messages:]
    

    # get user query
    if user_query := st.chat_input("Enter your query here...", key="user_query"):
        
        # display user query
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Generate AI response based on user query
        with st.chat_message("AI", avatar="🦜"):
            # if model_option == "python": 
            if option == "in-class":       
                ai_response = st.write_stream(
                    chat_chain.stream(input={'query': user_query, 
                                               'chat_history': st.session_state.chat_history}))

            if option == "assignment":       
                ai_response = st.write_stream(
                    step_chain.stream(input=user_query))
                
            # model option is RAG for the course    # 
            else:                
                ai_response = st.write_stream(
                    rag_chain.stream(input=user_query))

        # append AI response to chat history
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(ai_response))

    # summarize the conversation with llm. 
if __name__ == '__main__':
    main()
