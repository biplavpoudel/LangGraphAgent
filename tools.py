import csv
from typing import List, Dict

import requests
import os
import pytesseract
import base64
import yt_dlp
from PIL import Image
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from duckduckgo_search import DDGS
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.utilities import SearxSearchWrapper

from dotenv import load_dotenv
from yt_dlp import YoutubeDL
import webvtt

load_dotenv()


@tool("add_tool", parse_docstring=False)
def add(a: float, b: float) -> float:
    """Add two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a + b


@tool("subtract_tool", parse_docstring=False)
def subtract(a: float, b: float) -> float:
    """Subtract a number from another.

    Args:
        a: First operand
        b: Second operand
    """
    return a - b


@tool("multiply_tool", parse_docstring=False)
def multiply(a: float, b: float) -> float:
    """Multiply two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a * b


@tool("division_tool", parse_docstring=False)
def divide(a: float, b: float) -> float:
    """Divide two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a / b


@tool("modulo_tool", parse_docstring=False)
def modulo(a: int, b: int) -> int:
    """Reminder of division of two numbers.

    Args:
        a: First operand
        b: Second operand
    """
    return a % b


@tool("power_tool", parse_docstring=False)
def power(a: float, b: float) -> float | complex:
    """Value of a raised to the power of b (i.e. a^b)

    Args:
        a: Base number
        b: Exponent number
    """
    return pow(a, b)


@tool("web_search", parse_docstring=False)
def web_search(query: str) -> str:
    """Search the keyword in Search Engine.

    Args:
        query (str): String to search for
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
        name (str): String to lookup in Wikipedia."""
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
        name (str): Name of paper to lookup in arXiv.org"""
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
        url (str): URL of the resource to download.

    Returns:
        (str) : The path of the downloaded file.
    """
    filename = url.split("/")[-1]
    download_dir = os.getcwd() + "/downloads/"
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, filename)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
    }
    print(f"Attempting to download from: {url}")
    print(f"Saving to: {file_path}")
    try:
        with requests.get(url, stream=True, headers=headers, timeout=5) as r:
            r.raise_for_status()
            expected_size = r.headers.get("content-length")
            if expected_size:
                expected_size = int(expected_size)
                print(f"Expected file size: {expected_size} bytes.")
            else:
                print("Content-Length header not found.")
            downloaded_size = 0
            with open(file_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # to remove keep-alive chunks
                        f.write(chunk)
                        downloaded_size += len(chunk)
            if expected_size and downloaded_size != expected_size:
                print(
                    f"WARNING: Downloaded size ({downloaded_size}) does not match expected size ({expected_size}). File might be incomplete."
                )

            print(f"File saved at: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error during download (HTTP or network issue): {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up partial download
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up partial download
        return ""
    return file_path


@tool("excel_loader", parse_docstring=True)
def excel_loader(path: str) -> str:
    """Returns Excel file as formatted string.

    Args:
        path (str): Path to Excel file"""
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
        path (str): Path to pdf file"""
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
        path (str): Path to csv file"""
    docs = []
    formatted_result = ""
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        header = next(reader, None)
        if header:
            # print(f"Header: {header}")
            formatted_result = f"<Document source={path}>, Header={header}, Content=<"
        for row in reader:
            docs.append([",".join(row)])
    formatted_result = formatted_result + ",".join([f"{page}" for page in docs])
    return str(formatted_result)


@tool("image_text_extractor", parse_docstring=True)
def image_text_extractor(path: str, model: str = "pytesseract") -> str:
    """Returns text from image file using OCR library pytesseract (if available).

    Args:
        path (str): Path to image file
        model (Optional): defaults to PyTesseract. For local use with Ollama, Gemma3 can be used
    """
    result = ""
    if model == "pytesseract":
        result = pytesseract.image_to_string(Image.open(path), output_type="string")
    elif model == "ollama":
        with open(path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

        # Send to Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b-it-q8_0",
                "options": {"temperature": 1, "top_p": 0.95, "top_k": 64},
                "prompt": "What is in this image?",
                "images": [img_base64],
                "stream": False,
            },
        )
        result = response.json()["response"]
    print(f"Text from image '{path.split('/')[-1]}' is :\n{result}")
    return result

@tool("youtube_caption_downloader", parse_docstring=True)
def youtube_caption_downloader(url: str, out_dir:str = "downloads", lang:List=['en'], sub_format:str='vtt') -> List[Dict]:
    """Alternative to transcribing the audio from YouTube videos.
    Returns YouTube caption downloaded from given URL.

    Args:
        url (str): YouTube URL
        out_dir (Optional): defaults to 'downloads'
        lang (List): defaults to ['en']
        sub_format (Optional): defaults to 'vtt'
    """
    download_dir = os.getcwd() + f"/{out_dir}/"
    os.makedirs(download_dir, exist_ok=True)
    success = False
    captions_list = []
    actual_path=""
    # Arguments for yt-dlp
    opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': lang,
        'subtitlesformat': sub_format,
        'skip_download': True,
        'outtmpl': os.path.join(download_dir, '%(title)s-%(id)s.%(ext)s'),
        'quiet': True,
        'warnings': False,
        'retries': 2
    }
    try:
        with YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', 'Unknown Title')
            video_id = info_dict.get('id', 'Unknown ID')
            print(f"Downloading caption for {video_title}({video_id}) from URL {url}")
            subtitles = info_dict.get('subtitles', {})
            auto_captions = info_dict.get('automatic_captions', {})
            if lang[0] in subtitles:
                print(f"Manual subtitles found and downloaded for language: {lang[0]}")
            elif lang[0] in auto_captions:
                print(f"Automatic subtitles found and downloaded for language: {lang[0]}")
            else:
                print(f"No subtitles available for language '{lang[0]}'")

            # Estimate subtitle filename
            subtitle_filename1 = f"{video_title}-{video_id}.{lang[0]}.vtt"
            subtitle_filename2 = f"{video_title}-{video_id}.{lang[0]}.srt"
            subtitle_path1 = os.path.join(download_dir, subtitle_filename1)
            subtitle_path2 = os.path.join(download_dir, subtitle_filename2)

            if os.path.exists(subtitle_path1):
                print(f"Subtitle saved at: {subtitle_path1}")
                actual_path = subtitle_path1
            elif os.path.exists(subtitle_path2):
                print(f"Subtitle saved at: {subtitle_path2}")
                actual_path = subtitle_path2
            else:
                print("Subtitle file was not found in the expected path.")
            success = True
    except yt_dlp.DownloadError as e:
        print("Download error:", e)
    except yt_dlp.SameFileError as e:
        print("Same file error:", e)
    except Exception as e:
        print("Unknown error:", e)

    # Now reading the vtt file and returning it as result string
    if success:
        for i, caption in enumerate(webvtt.read(actual_path), 1):
            caption_dict = {
                "caption_id": caption.identifier or f"cue_{i}",
                "caption_start": caption.start,
                "caption_end": caption.end,
                "caption_text": caption.text.strip(),
                "caption_voice": caption.voice or None
            }
            captions_list.append(caption_dict)
    return captions_list


if __name__ == "__main__":
    keyword = "Attention is all you need"
    # print(wiki_search.invoke(keyword))
    # print(web_search.invoke(keyword))
    # print(arxiv_search.invoke(keyword))
    print(youtube_caption_downloader.invoke("https://www.youtube.com/watch?v=1htKBjuUWec"))
