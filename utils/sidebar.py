import streamlit as st
from datetime import datetime

def clear_chat_history():
    """Clear the chat history and reset conversation."""
    st.session_state.chat_history = []
    st.session_state.memory_summary = ""
    st.session_state.last_interaction_id = ""
    st.session_state.pending_intent = None
    st.rerun()

def save_chat_history():
    """Save chat history in a readable format."""
    if 'chat_history' not in st.session_state or not st.session_state.chat_history:
        return "No conversation history to save."
    
    # Create formatted chat history
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    chat_content = f"DDA Virtual TA - Chat History\n"
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

def sidebar():
    """Enhanced sidebar for MBA Data Analytics Virtual TA."""
    with st.sidebar:
        
        # Header and branding
        st.markdown("# Virtual TA Guide")
        # st.markdown("**ISOM 550 - Data and Decision Analytics**")
        # st.markdown("---")

        st.markdown("## Learning Preferences")
        response_modes = ["Direct answer", "Hint-first", "Teach me step-by-step"]
        if "response_mode" not in st.session_state:
            st.session_state.response_mode = "Teach me step-by-step"
        response_mode = st.segmented_control(
            "Response style",
            options=response_modes,
            key="response_mode",
            help="Choose how much guidance the assistant should provide.",
        )
        if response_mode is None:
            response_mode = "Teach me step-by-step"
            st.session_state.response_mode = response_mode
        memory_window = st.slider(
            "Recent message window",
            min_value=4,
            max_value=16,
            value=st.session_state.get("memory_window", 8),
            step=2,
            help="Number of recent messages considered before using summary memory.",
        )
        show_diagnostics = st.toggle(
            "Show agent diagnostics",
            value=st.session_state.get("show_diagnostics", False),
            help="Display tool calls and retrieval debug info for development.",
        )
        st.session_state.memory_window = memory_window
        st.session_state.show_diagnostics = show_diagnostics

        # Conversation management
        st.markdown("## Conversation")
        # Show conversation count
        if 'chat_history' in st.session_state:
            msg_count = len(st.session_state.chat_history)
            st.markdown(f"**Messages:** {msg_count}")
        
        # Conversation management buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Clear chat button
            if st.button("🗑️ Clear Chat", width='stretch'):
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
                    width='stretch'
                )
            else:
                st.button("💾 Save Chat", disabled=True, width='stretch', 
                         help="No conversation to save")
        
        st.markdown("---")
        
        # Pro tips
        # st.markdown("## 🎯 Pro Tips")

        # App behavior explanation
        st.markdown("## How It Works")
        
        with st.expander("Unified Tutor", expanded=False):
            st.markdown("""
            **Auto-routing:**
            - Course logistics questions route to course materials
            - Learning and assignment questions route to content materials
            - If routing fails, the app falls back to direct tutoring safely
            
            **💡 Example Queries:**
            - "When is the midterm exam?"
            - "Help with Assignment 3"
            - "What's the grading policy?"
            - "Guide me through regression analysis"
            - "What materials do I need for the final project?"
            - "How do I interpret this statistical output?"
            """)
        
        # st.markdown("---")
        
        with st.expander("Get Better Results", expanded=False):
            st.markdown("""
            **Be Specific:**
            - Include context and variables
            - Specify your dataset/scenario
            - Ask follow-up questions
            
            **Conversation Flow:**
            - Build on previous responses
            - Ask "What's next?" for step-by-step help
            - Reference earlier discussion
            
            **Learning Tip:**
            - Ask focused follow-up questions to deepen understanding
            - Mention your assumptions so the tutor can correct them early
            """)
        
        
        
        # Footer
        st.markdown("## Important Notes")
        st.markdown("""
        **Work in Progress:** Continuously improving
        
        **Privacy:** Never include personal information
        
        **Academic Tool:** Use for learning, not cheating
        
        **Learning Focus:** Ask for hints when you want guided practice
        """)
        
        st.markdown("---")
        st.markdown("**Created by:** Dr. Wenjun Gu")  
        st.markdown("📧 wenjun.gu@emory.edu")
        st.markdown("🏫 Goizueta Business School")

    return {
        "response_mode": response_mode,
        "memory_window": memory_window,
        "show_diagnostics": show_diagnostics,
    }

def update_session_stats():
    """Update session statistics (call from main app)."""
    if 'total_queries' in st.session_state:
        st.session_state.total_queries += 1
    else:
        st.session_state.total_queries = 1

def get_sidebar_settings():
    return {
        "response_mode": st.session_state.get("response_mode", "Teach me step-by-step"),
        "memory_window": st.session_state.get("memory_window", 8),
        "show_diagnostics": st.session_state.get("show_diagnostics", False),
    }
