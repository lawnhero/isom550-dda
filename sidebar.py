import streamlit as st
import os

def clear_chat_history():
    st.session_state.chat_history = []

def sidebar():
    with st.sidebar:
        
        st.markdown(
            "## Example use cases\n"
            "1. How can I contact the professor?\n" 
            "2. What's a parameter in Python function\n"
            "3. How to use my own module in Colab? \n"
            "4. How to fix the error in my code? \n"
            "5. Give me a practice question on Python functions\n"
        )
        # st.markdown("---")
        st.markdown(
            "## General advice\n"
            "1. Provide detailed context.\n" 
            "2. Try your questions in different ways.\n"
            "3. Query about specific task.\n"
        )
        st.markdown("---")
        st.markdown("# About")
        st.markdown(
            '''📖 Virtual TA allows you to ask questions about course logistics, 
            as well as any Python coding questions.'''
        )
        st.markdown(
            "This tool is a work in progress. "
        )
        st.markdown("Made by Dr.Wenjun Gu (wenjun.gu@emory.edu)")
       

        # api_key_input = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     placeholder="Paste your OpenAI API key here (sk-...)",
        #     help="You can get your API key from https://platform.openai.com/account/api-keys.",  # noqa: E501
        #     value=os.environ.get("OPENAI_API_KEY", None)
        #     or st.session_state.get("OPENAI_API_KEY", ""),
        # )

        # user_name = st.text_input(
        #     "Enter your name ",
        #     # type="password",
        #     placeholder="Your prefered name here...",
        #     help="This information will be deleted after each session",  # noqa: E501
        #     value='ISOM 352 Coder',
        # )

        # st.session_state["user_name"] = user_name

#         st.markdown("---")


#         st.markdown(
#         """
# # FAQ
# ## Best practice for interacting with virtual TA?
# Your query should provide as much information as possible to help identify the correct knowledge.  
# """
#         )
