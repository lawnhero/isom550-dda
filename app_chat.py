import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.globals import set_verbose

import utils.chains_lcel as chains
from utils.sidebar import sidebar

# Enable verbose logging
set_verbose(True)

# Set the page_title
st.set_page_config(page_title="🦜 GBS ISOM 550 DDA Virtual TA - Beta", page_icon="🔍")
# st.set_page_config(page_title="Streamlit App", page_icon=":shark:", layout="wide")

# cache the vectorized embedding database 
@st.cache_resource
# load the vectorized database
def load_db(db_path, embedding_model='text-embedding-ada-002'):
    embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=1)
    db_loaded = FAISS.load_local(db_path, embeddings)
    
    return db_loaded

# 1. Load the Vectorised database
kb_db_path = 'data/emb_db'
db = load_db(kb_db_path)

# 2. Function for similarity search
retriever = db.as_retriever()

# 3. Setup LLM and chains
llm = ChatOpenAI(temperature=0.1, 
                #  model="gpt-4-0125-preview",
                 model="gpt-3.5-turbo-1106",
                 verbose=True,
                 max_tokens=300,
                 )
# replace query router with streamlit toggle button

# 3a. Setup query router
# router_chain = chains.router_chain(llm)

# def router_choice(query, chain):
#     choice = chain.invoke(input={'query': query})
#     return int(choice)

# 3b. Setup LLMChain & prompts for RAG answer generation
rag_chain = chains.rag_chain(llm, retriever)

# 3c. Setup direct openai_chain
openai_chain = chains.openai_chain(llm)

# 3d. Setup chat history chain
chat_history_chain = chains.chat_history_chain(llm)

# 4. generate response based on model choice

def generate_response(query_inputs, model_selection):
    # query_inputs keys: "query", "chat_history"
    # if choice == 0: # LLM decides to use chat history
    #     decision = "use chat history"
    #     st.markdown(f"🦜Virtual TA: I'm going to {decision} 🐍")
    #     # Generate response with chat history last 2 messages
    #     response = chat_history_chain.stream(input=query_inputs)
    # 
    if 'python' in model_selection: # LLM decides to use OpenAI directly
        decision = "use OpenAI directly"
        st.markdown(f"🦜Virtual TA: I'm going to {decision} 🐍")
        # Generate response with chat history last 2 messages
        response = openai_chain.stream(input=query_inputs)

    else: 
        decision = "get more information" # LLM router to RAG
        st.markdown(f"🦜Virtual TA: I need to {decision} 🔍")
        # with st.spinner(f"Generating answers..."): 
        response = rag_chain.stream(input=query_inputs['query'])
    
    return response



# 5. Build an app with streamlit
def main():

    st.title("ISOM 550 Virtual TA - Beta 🔍")
    sidebar()

    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # truncate chat history to last 5 messages
    if len(st.session_state.chat_history) > 5:
        st.session_state.chat_history = st.session_state.chat_history[-5:]

    # Create a toggle button to choose between Python and Course
    model_option = (
        "course" if st.toggle("Query on Python ⇄ Course", value=False) else "python"
    )

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
    
    # if model_option == "python": 
    if model_option == "python":
        # get user query
        if user_query := st.chat_input("Hello there. Feel free to ask me Python? 🐍"):
            st.session_state.chat_history.append(HumanMessage(user_query))
            # display user query
            with st.chat_message("Human"):
                st.markdown(user_query)
                
            # st.markdown(f"🦜Virtual TA: I'm going to OpenAI 🐍")
            # get response from AI
            with st.chat_message("AI"):
                ai_response = st.write_stream(
                    openai_chain.stream(input={'query': user_query, 
                    'chat_history': st.session_state.chat_history}))
            # append AI response to chat history
            st.session_state.chat_history.append(AIMessage(ai_response))
    
    # model option is RAG for the course    # 
    else:
        if user_query := st.chat_input("Hello there. Ask me about the Course? 🔍"):
            st.session_state.chat_history.append(HumanMessage(user_query))
            # display user query
            with st.chat_message("Human"):
                st.markdown(user_query)

            # st.markdown(f"🦜Virtual TA: I'm going to retrieve data ")
            # get response from AI
            with st.chat_message("AI"):
                ai_response = st.write_stream(rag_chain.stream(input=user_query))

            # append AI response to chat history
            st.session_state.chat_history.append(AIMessage(ai_response))
    

if __name__ == '__main__':
    main()