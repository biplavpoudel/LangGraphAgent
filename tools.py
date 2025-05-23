from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_community.docstore import Wikipedia
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from duckduckgo_search import DDGS
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader
import csv
import requests
import os
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

@tool("file_downloader", parse_docstring=False)
def file_downloader(url: str) -> str:
    """Downloads file from the input url.
    Args:
        url : URL of the resource to download.
    Returns:
        (str) : The path of the downloaded file.
        """
    filename = url.split('/')[-1]
    download_dir = os.getcwd()+ "/downloads/"
    file_path = os.path.join(download_dir, filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return file_path

@tool("excel_loader", parse_docstring=True)
def excel_loader(path: str) -> str:
    """Returns Excel file as formatted string.
    Args:
        path : Path to Excel file"""
    loader = UnstructuredExcelLoader(file_path=path, mode="elements")
    docs = loader.load()
    formatted_result = f"\n\n{'-' * 50}\n\n".join(
        [
            f"<Document Title={page.metadata['page_name']}, Page={page.metadata['page_number']}'>\n{page.metadata['text_as_html'] if page.metadata['category'] == 'Table' else {page.page_content}}"
            for page in docs
        ]
    )
    return str(formatted_result)

@tool("pdf_loader", parse_docstring=True)
def pdf_loader(path: str) -> str:
    """Returns pdf file as formatted string.
    Args:
        path : Path to pdf file"""
    loader = PyPDFLoader(file_path=path, mode="single")
    docs = loader.load()
    formatted_result = f"\n\n{'-' * 50}\n\n".join(
        [
            f"<Document Source={page.metadata['source']}, Title={page.metadata['title']}, Authors={page.metadata['author']}'>\n{page.page_content[:1000]}"
            for page in docs
        ]
    )
    return str(formatted_result)

@tool("csv_loader", parse_docstring=True)
def csv_loader(path: str) -> str:
    """Returns pdf file as formatted string.
    Args:
        path : Path to csv file"""
    docs = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        header = next(reader, None)
        if header:
            # print(f"Header: {header}")
            formatted_result = f"<Document source={path}>, Header={header}, Content=<"
        for row in reader:
            docs.append([','.join(row)])
    formatted_result = formatted_result + ",".join(
        [
            f"{page}"
            for page in docs
        ]
    )
    return str(formatted_result)

if __name__ == "__main__":
    keyword = "Attention is all you need"
    # print(wiki_search.invoke(keyword))
    # print(web_search.invoke(keyword))
    print(arxiv_search.invoke(keyword))
