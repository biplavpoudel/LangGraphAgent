from pprint import pprint
from typing import Literal

from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, MessagesState
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from tools import add, modulo, divide, multiply, subtract, power, arxiv_search, web_search, wiki_search, file_downloader, excel_loader, csv_loader, pdf_loader, image_text_extractor
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_core.documents import Document
from tool_logger import ToolLogger
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()


def build_graph(llm_provider: str = "gemma"):
    # Adding LLM/VLM Model
    if llm_provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
                max_new_tokens=1024,
                do_sample=False,
                repetition_penalty=1.03,
                temperature=0,
            ),
            verbose=True,
        )
    elif llm_provider == "gemma":
        llm = ChatGoogleGenerativeAI(model="gemini/gemini-2.0-flash-lite-001")
    elif llm_provider == "ollama":
        # httpx.ConnectError when Ollama not running in background
        llm = ChatOllama(model="qwen3:4b", temperature=0.1)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_provider}")

    # Let's add system prompt from system_prompt.txt
    with open("system_prompt.txt", encoding="utf-8") as f:
        sys_message = f.read()
    system_prompt = SystemMessage(content=sys_message)

    # Adding Embedding Model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Adding Vector Store
    vector_store = InMemoryVectorStore(embeddings)
    # client = QdrantClient(url="https://e75c4d7b-d8c8-4f7b-9e54-892bc1583cb9.europe-west3-0.gcp.cloud.qdrant.io:6333",
    #                       api_key=os.environ["QDRANT_API_KEY"])
    # client.create_collection(
    #     collection_name="langchain_assistant",
    #     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    # )
    # vector_store = QdrantVectorStore(
    #     client=client,
    #     collection_name="langchain_assistant",
    #     embedding=embeddings,
    # )

    # Initialize Logger
    logger = ToolLogger(vector_store)

    # wrapping tools with logger so the result from tools are stored in vectordb
    web_search_logged = logger.wrap(web_search)
    wiki_search_logged = logger.wrap(wiki_search)
    arxiv_search_logged = logger.wrap(arxiv_search)
    excel_loader_logged = logger.wrap(excel_loader)
    pdf_loader_logged = logger.wrap(pdf_loader)
    csv_loader_logged = logger.wrap(csv_loader)

    # List of Tools that are bound to the LLM
    tools = [
        add,
        subtract,
        multiply,
        divide,
        modulo,
        arxiv_search,
        web_search,
        wiki_search,
        power,
        file_downloader,
        excel_loader,
        pdf_loader,
        csv_loader,
        image_text_extractor,
    ]
    logged_tools = [
        add,
        subtract,
        multiply,
        divide,
        modulo,
        power,
        arxiv_search_logged,
        web_search_logged,
        wiki_search_logged,
        file_downloader,
        excel_loader_logged,
        pdf_loader_logged,
        csv_loader_logged,
        image_text_extractor,
    ]
    print("Binding tools with the LLM....")
    llm_with_tools = llm.bind_tools(tools)

    graph_builder = StateGraph(MessagesState)
    retriever = vector_store.as_retriever()

    # Creating Node
    def assistant(state: MessagesState):
        all_messages_for_llm = [system_prompt] + state["messages"]
        response_message = llm_with_tools.invoke(all_messages_for_llm)
        print(f"\nDEBUG: LLM Raw Response Type: {type(response_message)}")
        print(f"DEBUG: LLM Raw Response Content: {response_message.content}")
        if isinstance(response_message, AIMessage) and response_message.tool_calls:
            print(f"DEBUG: LLM identified tool calls: {response_message.tool_calls}")
        else:
            print("DEBUG: LLM did NOT identify tool calls.")
        return {"messages": [response_message]}

        # return {
        #     "messages": [llm_with_tools.invoke(state["messages"] + [system_prompt])],
        # }

    def retrieve_documents(state: MessagesState):
        latest_user_msg = [
            msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)
        ][-1]
        retrieved_docs = retriever.invoke(latest_user_msg)
        return {
            "messages": [
                SystemMessage(
                    content="\n\n".join([doc.content for doc in retrieved_docs])
                )
            ]
        }

    # Add Nodes to Graph
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_node("document_retriever", retrieve_documents)

    # Add Edges to Graph
    graph_builder.add_edge(START, "document_retriever")
    graph_builder.add_edge("document_retriever", "assistant")
    graph_builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    graph_builder.add_edge("tools", "assistant")

    # Build Graph
    return graph_builder.compile()


if __name__ == "__main__":
    # question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # question = "Search Wikipedia for 'Python programming language'"
    question = "What is 200 power 8.99999?"
    graph = build_graph(llm_provider="ollama")

    messages = [HumanMessage(content=question)]
    response = graph.invoke({"messages": messages})
    answer = response["messages"][-1].content
    print(answer)
    # for m in response["messages"]:
    #     pprint(m)
