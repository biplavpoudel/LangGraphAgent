from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_community.docstore import Wikipedia
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from duckduckgo_search import DDGS
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
# from langchain_community.utilities import SearxSearchWrapper

from dotenv import load_dotenv

load_dotenv()


@tool("add_tool", parse_docstring=True)
def add(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a + b


@tool("subtract_tool", parse_docstring=True)
def subtract(a: int, b: int) -> int:
    """Subtract a number from another.

    Args:
        a: First operand
        b: Second operand
    """
    return a - b


@tool("multiply_tool", parse_docstring=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b


@tool("division_tool", parse_docstring=True)
def divide(a: int, b: int) -> float:
    """Divide two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a / b


@tool("modulo_tool", parse_docstring=True)
def modulo(a: int, b: int) -> int:
    """Reminder of division of two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a % b


@tool("web_search", parse_docstring=False)
def web_search(query: str) -> str:
    """Search the keyword in Search Engine.

    Args:
        query: String to search for
    """
    try:
        results = DDGS().text(keywords=query, max_results=2)
        formatted_result = f"\n\n{'-' * 50}\n\n".join(
            [
                f"<Document Source = '{page['href']}', Title = '{page['title']}'>\n {page['body']}"
                for page in results
            ]
        )
        return formatted_result
    except DuckDuckGoSearchException as e:
        print(
            f"Exception occur with DuckDuckGo! {str(e)}. \n Trying another search engine "
        )

    try:
        search_tool = TavilySearch(max_results=2, topic="general")
        result = search_tool.invoke(f'"query":"{query}"')["results"]
        formatted_result = f"\n\n{'-' * 50}\n\n".join(
            [
                f"<Document Source = '{page['url']}', Title = '{page['title']}'>\n {page['content']}"
                for page in result
            ]
        )
        return formatted_result

    except Exception as e:
        print(f"Exception occur with Tavily! {str(e)}. \n Returning empty json")
        return ""


@tool("wiki_search", parse_docstring=False)
def wiki_search(name: str) -> str:
    """Search the keyword in Wikipedia.
    Args:
        name : String to lookup in Wikipedia."""
    wikipedia_result = WikipediaLoader(
        query=name, load_max_docs=2, load_all_available_meta=True
    ).load()
    formatted_result = f"\n\n{'-' * 50}\n\n".join(
        [
            f"<Document Source = '{page.metadata['source']}'>\n{page.page_content}"
            for page in wikipedia_result
        ]
    )
    return str(formatted_result)


@tool("arxiv_search", parse_docstring=False)
def arxiv_search(name: str) -> str:
    """Search the paper in arXiv.org
    Args:
        name : Name of paper to lookup in arXiv.org"""
    arxiv_result = ArxivLoader(
        query=name, load_max_docs=2, load_all_available_meta=True
    ).load()
    formatted_result = f"\n\n{'-' * 50}\n\n".join(
        [
            f"<Document Title={doc.metadata['Title']}, Published={doc.metadata['Published']}, Authors={doc.metadata['Authors']}'>\n{doc.page_content[:5000]}"
            for doc in arxiv_result
        ]
    )
    return str(formatted_result)


if __name__ == "__main__":
    keyword = "Attention is all you need"
    # print(wiki_search.invoke(keyword))
    # print(web_search.invoke(keyword))
    print(arxiv_search.invoke(keyword))
