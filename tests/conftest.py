"""Shared fixtures for the pure-Python parts of the tutor.

Everything here runs without API keys, a vector index, or a Streamlit runtime.
Streamlit is imported transitively (utils.ui, utils.course_context) and prints
"No runtime found" warnings, which is expected and harmless.

Run from the repo root:

    python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def chat_history():
    return [
        AIMessage("Hi, I'm Dayton, your virtual TA."),
        HumanMessage("What does an R-squared of 0.62 mean?"),
        AIMessage("An R-squared of 0.62 means 62% of the variation is explained."),
    ]
