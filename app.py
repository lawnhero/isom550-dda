import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.globals import set_verbose
import utils.chains_lcel as chains
from utils.sidebar import sidebar, update_session_stats, set_current_mode
# from utils.llm_models import LLMModels
import utils.llm_models as llms


# Enable verbose logging
set_verbose(True)

# Set the page_title
st.set_page_config(
        page_title="ISOM 550 DDA Virtual TA - Beta", page_icon="🔍", layout="wide")

# cache the vectorized embedding database 
from utils.utils import load_db, query_db_connection, process_and_store_query

# 1. Load the Vectorised database
course_path = 'data/course'
contents_path = 'data/contents'
course_db = load_db(db_path=course_path)
contents_db = load_db(db_path=contents_path)
retriever_course = course_db.as_retriever()
retriever_contents = contents_db.as_retriever() 

# 2. MongoDB Atlas connection
mongo_db = query_db_connection()
collection = mongo_db['ISOM 550']

# 3. Setup LLM and chains
claude_sonnet = llms.claude_sonnet_with_fallback
claude_haiku = llms.claude_haiku_with_fallback
gpt4o = llms.openai_gpt4o
router_llm = llms.openai_gpt4o_mini  # Use JSON-capable model for routing

# Initialize all chains including the routing chain with specialized router LLM
all_chains = chains.get_all_chains(claude_sonnet, claude_haiku, retriever_course, retriever_contents, router_llm)
router = all_chains['router']
chain_dict = all_chains['chain_dict']

# Individual chains
step_chain = all_chains['step_chain']
rag_chain = all_chains['rag_chain']  
class_chain = all_chains['class_chain']
        
# 4. Build an app with streamlit
def main():

    st.header("🦜 Virtual TA - ISOM 550 DDA")
    sidebar()  # Enhanced sidebar with app functionality
    
    # Set up the radio button toggle with two options
    option = st.radio(
        label="Choose your interaction type:",
        options=["In-Class", "Course"],
        index=1,  # Default to Course
        horizontal=True,
        help="In-Class: General analytics discussions and explanations. Course: Course materials and assignment guidance."
    ).lower()

    # Update current mode in session state for sidebar
    set_current_mode(option.title())

    # Set initial message based on option
    if "in-class" in option:
        initial_text = "I can explain concepts, create practice problems, or help with software implementation."
        st.write("💡 **In-Class Mode**: Ask me about data analytics concepts, request practice problems, or get help with software implementation.")
    else:
        initial_text = "I can search course materials for logistics or provide step-by-step guidance for assignments."
        st.write("📚 **Course Mode**: Search course logistics (syllabus, deadlines) or provide learning guidance (concepts, assignments).")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(initial_text)]

    # Display conversation statistics in main area
    if len(st.session_state.chat_history) > 1:
        st.caption(f"💬 Conversation: {len(st.session_state.chat_history)} messages")

    # display previous conversation history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🦜"):
                st.markdown(message.content)
    
    # truncate chat history to last 4 messages to maintain context
    max_num_messages = 4
    if len(st.session_state.chat_history) > max_num_messages:
        st.session_state.chat_history = st.session_state.chat_history[-max_num_messages:]
    

    # get user query
    if user_query := st.chat_input("Enter your query here...", key="user_query"):
        
        # Update session statistics
        update_session_stats()
        
        # display user query
        with st.chat_message("Human"):
            st.markdown(user_query)
        
        # save to MongoDB database
        process_and_store_query(collection, query=user_query)

        # Generate AI response based on selected option
        with st.chat_message("AI", avatar="🦜"):
            try:
                if "in-class" in option:
                    # In-Class: Use class chain directly for general analytics discussions
                    ai_response = st.write_stream(
                        class_chain.stream({
                            'query': user_query, 
                            'chat_history': st.session_state.chat_history
                        }))
                
                else:
                    # Course: Use router to choose between course logistics and content materials
                    
                    # Get conversation context for the router
                    if len(st.session_state.chat_history) >= 4:
                        recent_history = st.session_state.chat_history[-4:]
                        history_text = "\n".join([f"{'Human' if i % 2 == 0 else 'AI'}: {msg.content}" for i, msg in enumerate(recent_history)])
                    else:
                        history_text = "No previous conversation"
                    
                    tool_choice = router.invoke({
                        "query": user_query,
                        "chat_history": history_text
                    })

                    print(tool_choice)
                    
                    # Show router decision in sidebar (for debugging/transparency)
                    if hasattr(tool_choice, 'label'):
                        st.sidebar.caption(f"🤖 Router Decision: {tool_choice.label}")
                    
                    # Execute the appropriate chain based on routing decision
                    response_stream = chains.call_function(
                        tool_name=tool_choice.label,
                        query=tool_choice.query,
                        chains_dict=chain_dict,
                        chat_history=st.session_state.chat_history
                    )
                    
                    # Stream the response
                    ai_response = st.write_stream(response_stream)
                        
            except Exception as e:
                print(e)
                # Fallback to class chain for any errors
                st.write("Let me help you with that...")
                ai_response = st.write_stream(
                    class_chain.stream({
                        'query': user_query, 
                        'chat_history': st.session_state.chat_history
                    }))

        # append AI response to chat history
        st.session_state.chat_history.append(HumanMessage(user_query))
        st.session_state.chat_history.append(AIMessage(ai_response))

if __name__ == '__main__':
    main()
