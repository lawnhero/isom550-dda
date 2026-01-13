import streamlit as st
import os
from datetime import datetime
import json

def clear_chat_history():
    """Clear the chat history and reset conversation."""
    st.session_state.chat_history = []
    st.rerun()

def save_chat_history():
    """Save chat history in a readable format."""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return "No conversation history to save."
    
    # Create formatted chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_content = f"ISOM 550 DDA Virtual TA - Chat History\n"
    chat_content += f"Saved on: {timestamp}\n"
    chat_content += f"Total Messages: {len(st.session_state.chat_history)}\n"
    chat_content += "=" * 50 + "\n\n"
    
    for i, message in enumerate(st.session_state.chat_history, 1):
        if hasattr(message, 'content'):
            if "AI" in str(type(message)) or "Assistant" in str(type(message)):
                chat_content += f"🤖 AI Assistant:\n{message.content}\n\n"
            else:
                chat_content += f"👤 Student:\n{message.content}\n\n"
        chat_content += "-" * 30 + "\n\n"
    
    return chat_content

def get_mode_info():
    """Get information about current interaction mode."""
    # This will be called from the main app to show current mode
    return st.session_state.get('current_mode', 'Unknown')

def sidebar():
    """Enhanced sidebar for MBA Data Analytics Virtual TA."""
    with st.sidebar:
        
        # Header and branding
        st.markdown("# 🤖 Virtual TA Guide")
        st.markdown("**BUS 350 - Data & Decision Analytics**")
        st.markdown("---")
        
        
        # Conversation management
        st.markdown("## 🔄 Conversation")
        # Show current mode if available
        if hasattr(st.session_state, 'current_mode'):
            st.markdown(f"**Current Mode:** {st.session_state.current_mode}")
        
        # Show conversation count
        if 'chat_history' in st.session_state:
            msg_count = len(st.session_state.chat_history)
            st.markdown(f"**Messages:** {msg_count}")
        
        # Conversation management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Clear chat button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat_history()
                st.success("Chat cleared!")
        
        with col2:
            # Save chat button
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                chat_content = save_chat_history()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.txt"
                
                st.download_button(
                    label="💾 Save Chat",
                    data=chat_content,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.button("💾 Save Chat", disabled=True, use_container_width=True, 
                         help="No conversation to save")
        
        st.markdown("---")
        
        # Pro tips
        # st.markdown("## 🎯 Pro Tips")

        # Mode explanation
        st.markdown("## 📋 How It Works")
        
        with st.expander("🔍 **In-Class Mode**", expanded=False):
            st.markdown("""
            **Best for:**
            - Concept explanations
            - Practice problems  
            - Software implementation help
            - General analytics discussions
            
            **AI Response:** Direct answers with examples, practice questions, or step-by-step guidance
            
            **💡 Example Queries:**
            - "Explain linear regression"
            - "Create practice questions on hypothesis testing"  
            - "What's the business meaning of correlation?"
            - "What's the difference between mean and median?"
            """)
        
        with st.expander("📚 **Course Mode**", expanded=False):
            st.markdown("""
            **Intelligent Routing:**
            - **Course Logistics** → Syllabus, deadlines, policies
            - **Learning Content** → Concepts, assignment help
            
            **AI Magic:** Automatically determines which knowledge base to search based on your question!
            
            **💡 Example Queries:**
            - "When is the midterm exam?"
            - "Help with Assignment 3"
            - "What's the grading policy?"
            - "Guide me through regression analysis"
            - "What materials do I need for the final project?"
            - "How do I interpret this statistical output?"
            """)
        
        # st.markdown("---")
        
        with st.expander("📈 **Get Better Results**", expanded=False):
            st.markdown("""
            **Be Specific:**
            - Include context and variables
            - Specify your dataset/scenario
            - Ask follow-up questions
            
            **Conversation Flow:**
            - Build on previous responses
            - Ask "What's next?" for step-by-step help
            - Reference earlier discussion
            
            **Course Mode:**
            - Let AI auto-route your questions
            - Mix logistics and learning queries
            - Trust the intelligent classification
            """)
        
        
        
        st.markdown("---")
        
        # Footer
        st.markdown("## ⚠️ Important Notes")
        st.markdown("""
        📖 **Work in Progress:** Continuously improving
        
        🔒 **Privacy:** Never include personal information
        
        🎓 **Academic Tool:** Use for learning, not cheating
        
        💬 **Feedback:** Report issues or suggestions
        """)
        
        st.markdown("---")
        st.markdown("**Created by:** Dr. Wenjun Gu")  
        st.markdown("📧 wenjun.gu@emory.edu")
        st.markdown("🏫 Goizueta Business School")

def update_session_stats():
    """Update session statistics (call from main app)."""
    if 'total_queries' in st.session_state:
        st.session_state.total_queries += 1
    else:
        st.session_state.total_queries = 1

def set_current_mode(mode):
    """Set current interaction mode (call from main app)."""
    st.session_state.current_mode = mode
