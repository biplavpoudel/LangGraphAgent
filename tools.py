import os
import json

from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_tavily import TavilySearch
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
def multiply(a: int, b: int) -> float:
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

@tool("web_search", parse_docstring=True)
def web_search(keyword: str) -> json:
    """Search the keyword in Search Engine.

    Args:
        keyword: String to search for
    """
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(region="en-us", max_results=2)
        search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="json")
        result = search.invoke(keyword)
        return result
    except DuckDuckGoSearchException as e:
        print(f"Exception occur with DuckDuckGo! {str(e)}. \n Trying another search engine ")

    try:
        search_tool = TavilySearch(
            max_results=2,
            topic="general")
        result = search_tool.invoke(f'"query":"{keyword}"')
        return result

    except Exception as  e:
        print(f"Exception occur with Tavily! {str(e)}. \n Returning empty json")
        return {}

@tool("wiki_search", parse_docstring=True)
def wiki_search(keyword: str) -> str:
    """Search the keyword in Wikipedia.
    Args:
        keyword: String to lookup in Wikipedia
    """



