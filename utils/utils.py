from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import streamlit as st

# knowledge base path
kb_db_path = 'data/emb_db'


@st.cache_resource
# load the vectorized database
def load_db(db_path=kb_db_path, embedding_model='text-embedding-ada-002'):
    embeddings = OpenAIEmbeddings(model=embedding_model, chunk_size=1)
    db_loaded = FAISS.load_local(db_path, embeddings, 
                                 allow_dangerous_deserialization=True
                                 )
    print("Database loaded")
    return db_loaded