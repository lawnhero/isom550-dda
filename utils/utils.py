import os
import certifi
import uuid
import chromadb
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


def open_chroma(persist_dir, embedding_function=None, embedding_model='text-embedding-3-small'):
    """Open one persist directory with its own Chroma client.

    chromadb.Client() is a process-wide singleton keyed by persist path, but
    LangChain's default constructor goes through that Client() helper and two
    indexes in one process (concepts + documents) end up sharing segment
    state. The documents query then dies in _decode_seq_id with
    `object of type 'int' has no len()`. PersistentClient(path=...) keeps
    each index isolated. Collection name stays `langchain` -- that is what
    the build scripts write.
    """
    persist = os.path.abspath(persist_dir)
    if embedding_function is None:
        embedding_function = OpenAIEmbeddings(model=embedding_model)
    client = chromadb.PersistentClient(
        path=persist,
        settings=Settings(anonymized_telemetry=False),
    )
    return Chroma(
        client=client,
        persist_directory=persist,
        embedding_function=embedding_function,
        collection_name="langchain",
    )


@st.cache_resource
# load the vectorized database
def load_db(db_path=kb_db_path, embedding_model='text-embedding-3-small', label=''):
    persist = os.path.abspath(db_path)
    db_loaded = open_chroma(persist, embedding_model=embedding_model)
    try:
        n = db_loaded._collection.count()
    except Exception as exc:
        n = f"unreadable ({exc})"
    print(f"Database loaded: {label} ({n} chunks from {persist})")
    return db_loaded

def _get_mongodb_uri() -> str:
    """Resolve MongoDB URI from Streamlit secrets (deployed) or MONGODB_URI in .env (local)."""
    try:
        secret_uri = st.secrets.get("mongodb_uri")
        if secret_uri:
            return secret_uri
    except Exception:
        pass

    env_uri = os.getenv("MONGODB_URI")
    if env_uri:
        return env_uri

    raise ValueError(
        "MongoDB URI not configured. Set mongodb_uri in .streamlit/secrets.toml "
        "or MONGODB_URI in .env."
    )


# MongoDB Atlas connection
@st.cache_resource
def query_db_connection():
    """Return a MongoDB connection to the user_queries_db database."""
    client = MongoClient(
        _get_mongodb_uri(),
        server_api=ServerApi("1"),
        tlsCAFile=certifi.where(),
    )
    print("Connected to MongoDB")
    return client['user_queries_db']


def build_event_payload(
    event_type: str,
    session_id: str,
    mode: str,
    response_mode: str,
    query: str = "",
    route_label: str = "",
    learning_objective: str = "",
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