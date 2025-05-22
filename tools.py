import os
import json

from PIL.features import features
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_community.docstore import Wikipedia
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_tavily import TavilySearch

# from langchain_community.utilities import SearxSearchWrapper
from langchain_community.document_loaders import WikipediaLoader

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
def add(a: int, b: int) -> int:
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
        wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", max_results=2)
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="json")
        result = search.invoke(query)
        # formatted_result = f"\n{'-' * 10}\n".join([
        #     f"<Document Source = '{page.metadata['source']}', Page = '{page.metadata['page']}'>\n {page.content}"
        #     for page in result])
        return result
    except DuckDuckGoSearchException as e:
        print(f"Exception occur with DuckDuckGo! {str(e)}. \n Trying another search engine ")
        return ""
    #
    # try:
    #     search_tool = TavilySearch(
    #         max_results=2,
    #         topic="general")
    #     result = search_tool.invoke(f'"query":"{query}"')
    #     # formatted_result = f"\n{'-' * 10}\n".join([
    #     #     f"<Document Source = '{page.metadata['source']}', Page = '{page.metadata['page']}'>\n {page.content}"
    #     #     for page in result])
    #     return result
    #
    # except Exception as  e:
    #     print(f"Exception occur with Tavily! {str(e)}. \n Returning empty json")
    #     return ""


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


if __name__ == "__main__":
    keyword = "Martin Luther King"
    # print(wiki_search.invoke(keyword))
    print(web_search.invoke(keyword))
    # print(divide.invoke({"a": 2, "b": 4}))
