import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]