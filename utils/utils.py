import os
import certifi
import uuid
from chromadb import Settings
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

# knowledge base path
kb_db_path = 'data/chroma_db'


@st.cache_resource
# load the vectorized database
def load_db(db_path=kb_db_path, embedding_model='text-embedding-3-small'):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    db_loaded = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        client_settings=Settings(anonymized_telemetry=False)
    )
    print("Database loaded")
    return db_loaded

MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

uri = f"mongodb+srv://streamlit_app:{MONGODB_PASSWORD}@virtual-ta.q344d.mongodb.net/?retryWrites=true&w=majority"

# MongoDB Atlas connection
@st.cache_resource
def query_db_connection():
    """Return a MongoDB connection to the user_queries_db database."""
    client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
    print("Connected to MongoDB")
    return client['user_queries_db']


# function to store the query in the database
def process_and_store_query(collection, **kwargs):
    """Insert a query into the MongoDB collection."""
    # Create a document to insert
    document = {
        "timestamp": datetime.now()
    }
    # add any additional fields to the document
    document.update(kwargs)
    
    # Insert the document into the collection
    result = collection.insert_one(document)
    
    return result.inserted_id


def build_event_payload(
    event_type: str,
    session_id: str,
    mode: str,
    response_mode: str,
    query: str = "",
    route_label: str = "",
    learning_objective: str = "",
    learner_level: str = "",
    resolved: bool = True,
    metadata: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Build a normalized event payload for analytics."""
    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": datetime.now(),
        "session_id": session_id,
        "mode": mode,
        "response_mode": response_mode,
        "query": query,
        "route_label": route_label,
        "learning_objective": learning_objective,
        "learner_level": learner_level,
        "resolved": resolved,
        "metadata": metadata or {},
    }
    return payload


def store_event(collection, payload: Dict[str, Any]):
    """Store event payload in MongoDB."""
    return collection.insert_one(payload).inserted_id


def store_feedback(
    collection,
    session_id: str,
    interaction_id: str,
    helpful: str,
    note: str = "",
    mode: str = "",
):
    """Store explicit student feedback events."""
    payload = build_event_payload(
        event_type="feedback",
        session_id=session_id,
        mode=mode,
        response_mode="n/a",
        resolved=helpful == "Helpful",
        metadata={
            "interaction_id": interaction_id,
            "helpful": helpful,
            "note": note,
        },
    )
    return store_event(collection, payload)