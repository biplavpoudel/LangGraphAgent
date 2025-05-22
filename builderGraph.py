import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from tools import add, modulo, divide, multiply, subtract, arxiv_search, web_search, wiki_search
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.documents import Document

from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

# Let's add system prompt from system_prompt.txt
with open("system_prompt.txt") as f:
    sys_message = f.read()
system_prompt = SystemMessage.from_text(sys_message)

# List of Tools that are later bound to the LLM
tools = [add, subtract, multiply, divide, modulo, arxiv_search, web_search, wiki_search]


class State(TypedDict):
    messages: List[AnyMessage]


def build_graph(llm_provider: str = 'gemma'):
    # Adding LLM/VLM Model
    if llm_provider == "huggingface":
        llm = HuggingFaceEndpoint(model='mistralai/Mistral-7B-Instruct-v0.3',
                                  # huggingfacehub_api_token=os.getenv('HUGGINGFACE_API'),
                                  verbose=True,
        )
    elif llm_provider == "gemma":
        llm = ChatGoogleGenerativeAI(model="gemini/gemini-2.0-flash-lite-001")
    elif llm_provider == "ollama":
        llm = ChatOllama(
            model="mistral:7b",
            temperature=0)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")
    llm_with_tools = llm.bind_tools(tools)

    # Adding Embedding Model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Adding Vector Store
    # vector_store = InMemoryVectorStore(embeddings)
    client = QdrantClient(url="https://e75c4d7b-d8c8-4f7b-9e54-892bc1583cb9.europe-west3-0.gcp.cloud.qdrant.io:6333", api_key=os.environ["QDRANT_API_KEY"])
    client.create_collection(
        collection_name="langchain_assistant",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="langchain_assistant",
        embedding=embeddings,
    )

    graph_builder = StateGraph()
    retriever = vector_store.as_retriever()

    # Creating Node
    def assistant(state: State):
        return {
            "message": [llm_with_tools.invoke(state["messages"])]
        }
    def retrieve_documents(state: State):
        latest_user_msg = [msg.content for msg in state["messages"] if isinstance(msg,HumanMessage)][-1]
        retrieved_docs = retriever.retrieve(latest_user_msg)
        return {
            "message": state["messages"] + [SystemMessage(content="\n\n".join([doc.content for doc in retrieved_docs ]))]
        }


    # Add Nodes to Graph
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("document_retriever", retrieve_documents)

    # Add Edges to Graph
    graph_builder.add_edge(START, "document_retriever")
    graph_builder.add_edge( "document_retriever", "assistant")
    graph_builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    graph_builder.add_edge("tools", "assistant")

    # Build Graph
    graph = graph_builder.compile()
