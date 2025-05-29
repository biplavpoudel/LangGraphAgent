import csv
import tempfile
from typing import List, Dict, Optional
import gc

import requests
import os
import pytesseract
import base64
import yt_dlp
import whisper
from PIL import Image
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from duckduckgo_search import DDGS
from langchain_community.document_loaders import WikipediaLoader, ArxivLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import PyPDFLoader
from chessimg2pos import predict_fen
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


@tool("string_reverse", parse_docstring=True)
def string_reverse(string: str) -> str:
    """Reverses the given string query.

    Args:
        string: Query string to be reversed
    """
    return string[::-1]


@tool("web_search", parse_docstring=False)
def web_search(query: str) -> str:
    """Search the keyword in Search Engine.

    Args:
        query (str): String to search for
    """
    try:
        results = DDGS().text(keywords=query, max_results=4)
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
        search_tool = TavilySearch(max_results=5, topic="general")
        result = search_tool.invoke({"query": query})
        pages = result.get("results", [])
        formatted_result = f"\n\n{'-' * 50}\n\n".join(
            [
                f"<Document Source = '{page['url']}', Title = '{page['title']}'>\n {page['content']}"
                for page in pages
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
        name (str): String to lookup in Wikipedia.
    """
    wikipedia_result = WikipediaLoader(
        query=name, load_max_docs=3, load_all_available_meta=True
    ).load()
    formatted_result = f"\n\n{'-' * 50}\n\n".join(
        [
            f"<Document Source = '{page.metadata['source']}'>\n{page.page_content}"
            for page in wikipedia_result
        ]
    )
    return str(formatted_result)


@tool("arxiv_search", parse_docstring=True)
def arxiv_search(name: str) -> str:
    """Search the paper in arXiv.org.

    Args:
        name (str): Name of paper to lookup in arXiv.org
    """
    arxiv_result = ArxivLoader(
        query=name, load_max_docs=3, load_all_available_meta=True
    ).load()
    formatted_result = f"\n\n{'-' * 50}\n\n".join(
        [
            f"<Document Title={doc.metadata['Title']}, Published={doc.metadata['Published']}, Authors={doc.metadata['Authors']}'>\n{doc.page_content[:5000]}"
            for doc in arxiv_result
        ]
    )
    return str(formatted_result)


@tool("file_downloader", parse_docstring=False)
def file_downloader(url: str, file_name: str = None) -> str:
    """Downloads file from the input url.
    Args:
        url (str): URL of the resource to download.
        file_name (str): Name of file to download.

    Returns:
        (str) : The path of the downloaded file."""
    filename = file_name if file_name else url.split("/")[-1]
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


@tool("read_code", parse_docstring=True)
def read_code(filepath: str) -> str:
    """Reads a programming language file such as Python, C, Java, etc.  and
    returns it as a string.

    Args:
        filepath (str): the path of the code file.
    """
    code = ""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'.")
        return None
    except IOError as e:
        print(f"Error reading file '{filepath}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing '{filepath}': {e}")

    return f"The code is returned as string: {code}."


@tool("save_to_file", parse_docstring=True)
def save_to_file(content: str, filename: Optional[str] = None) -> str:
    """Saves content to a file and returns the path.

    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random filename will be created.
    """
    download_dir = os.getcwd() + "/downloads/"
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=download_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(download_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to following directory: {filepath}."


@tool("excel_loader", parse_docstring=True)
def excel_loader(path: str) -> str:
    """Returns Excel file as formatted string.

    Args:
        path (str): Path to Excel file
    """
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
        path (str): Path to pdf file
    """
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
        path (str): Path to csv file
    """
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


@tool("image_analyzer", parse_docstring=True)
def image_analyzer(path: str, model: str = "ollama") -> str:
    """Analyzes image and returns text from image file.

    Args:
        path (str): Path to image file
        model (str): 'ollama' (default) or 'pytesseract'
    """

    if not os.path.exists(path):
        print(f"Error: Image file not found at '{path}'.")
        return None

    if model == "ollama":
        with open(path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        # Send to Ollama
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b-it-q8_0",
                "options": {"temperature": 1, "top_p": 0.95, "top_k": 64},
                "prompt": "Analyze the image properly and state what you found.",
                "images": [img_base64],
                "stream": False,
            },
        )
        result = response.json()["response"]
    elif model == "pytesseract":
        result = pytesseract.image_to_string(Image.open(path), output_type="string")

    print(f"Text from image '{path.split('/')[-1]}' is :\n{result}")
    return result


@tool("chess_positions", parse_docstring=True)
def chess_positions(path: str) -> str:
    """Returns FEN(Forsyth-Edwards Notation) positions of each chess pieces
    based on image provided. If the LLM is capable of Vision-related tasks,
    this tool will act as supplementary tool.

    Args:
        path (str): Path to chessboard image file.
    """
    result = predict_fen("downloads/cca530fc-4052-43b2-b130-b30968d8aa44.png")
    return result


@tool("audio_transcriber", parse_docstring=True)
def audio_transcriber(path: str, model: str = "whisper") -> str:
    """Returns audio transcript from given file.

    Args:
        path (str): Path to audio file
        model (Optional): Defaults to whisper by OpenAI with turbo model
    """
    result = ""
    try:
        if os.path.exists(path) and path.endswith(".mp3"):
            model = whisper.load_model("turbo")
            result = model.transcribe(model=model, audio=path)
            del model
            gc.collect()
            print(f"The audio transcript from the file {path} is: {result['text']}")
    except Exception as e:
        result = f"Couldn't transcribe the audio file: {str(e)}"
    return result


@tool("youtube_audio_downloader", parse_docstring=True)
def youtube_audio_downloader(
    url: str, out_dir: str = "downloads", audio_format: str = "mp3"
) -> str:
    """
    Downloads YouTube audio from given URL and returns path to downloaded file.
    Can be used for transcribing with another tool: audio_transcriber.
    Use it when the question doesn't need watching the YouTube video, just listening to audio and understanding the dialogue is enough.
    Not good when the task is to analyze each frame of video one-by-one.

    Alternative to: $ yt-dlp -x --audio-format mp3 -o "%(title)s-[%(id)s].%(ext)s" url

    Args:
        url (str): YouTube URL
        out_dir (str, Optional): defaults to 'downloads'
        audio_format (str, Optional): defaults to 'mp3'
    """
    download_dir = os.getcwd() + f"/{out_dir}/"
    os.makedirs(download_dir, exist_ok=True)
    # Arguments for yt-dlp
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
                "preferredquality": "192",  # You can modify this
            }
        ],
        "outtmpl": os.path.join(download_dir, "%(title)s-%(id)s.%(ext)s"),
        "quiet": True,
        "warnings": False,
        "retries": 2,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get("title", "Unknown Title")
            video_id = info_dict.get("id", "Unknown ID")
            print(f"Downloading audio for {video_title} ({video_id}) from URL {url}")
            expected_filename = os.path.join(
                download_dir, f"{video_title}-{video_id}.{audio_format}"
            )

            if os.path.exists(expected_filename):
                print(f"Audio saved at: {expected_filename}")
            else:
                print("Audio file was not found in the expected path.")
        return expected_filename
    except yt_dlp.DownloadError as e:
        print("Download error:", e)
    except yt_dlp.SameFileError as e:
        print("Same file error:", e)
    except Exception as e:
        print("Unknown error:", e)
    return f"Error downloading audio: {e}"


@tool("youtube_caption_downloader", parse_docstring=True)
def youtube_caption_downloader(
    url: str, out_dir: str = "downloads", lang: List = ["en"], sub_format: str = "vtt"
) -> List[Dict]:
    """Alternative to transcribing the audio from YouTube videos. Returns
    YouTube caption downloaded from given URL.

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
    actual_path = ""
    # Arguments for yt-dlp
    opts = {
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": lang,
        "subtitlesformat": sub_format,
        "skip_download": True,
        "outtmpl": os.path.join(download_dir, "%(title)s-%(id)s.%(ext)s"),
        "quiet": True,
        "warnings": False,
        "retries": 2,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get("title", "Unknown Title")
            video_id = info_dict.get("id", "Unknown ID")
            print(f"Downloading caption for {video_title}({video_id}) from URL {url}")
            subtitles = info_dict.get("subtitles", {})
            auto_captions = info_dict.get("automatic_captions", {})
            if lang[0] in subtitles:
                print(f"Manual subtitles found and downloaded for language: {lang[0]}")
            elif lang[0] in auto_captions:
                print(
                    f"Automatic subtitles found and downloaded for language: {lang[0]}"
                )
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
    # can be srt too, need another module pysrt (implement later)
    if success and actual_path[-3:] == "vtt":
        for i, caption in enumerate(webvtt.read(actual_path), 1):
            caption_dict = {
                "caption_id": caption.identifier or f"cue_{i}",
                "caption_start": caption.start,
                "caption_end": caption.end,
                "caption_text": caption.text.strip(),
                "caption_voice": caption.voice or None,
            }
            captions_list.append(caption_dict)
    elif not success:
        captions_list = [{"caption_id": f"Couldn't download caption: {str(e)}"}]
    return captions_list


if __name__ == "__main__":
    keyword = "Attention is all you need"
    # print(wiki_search.invoke(keyword))
    print(web_search.invoke(keyword))
    # print(arxiv_search.invoke(keyword))
    # # print(
    # #     youtube_caption_downloader.invoke("https://www.youtube.com/watch?v=1htKBjuUWec")
    # # )
    # # query = ".rewsna eht sa 'tfel' drow eht fo etisoppo eht etirw ,ecnetnes siht dnatsrednu uoy fI"
    # # print(string_reverse.invoke(query))
    # api_url = "https://agents-course-unit4-scoring.hf.space"
    # url = api_url + "/files/" + "7bd855d8-463d-4ed5-93ca-5fe35145f733"
    # file_name = "7bd855d8-463d-4ed5-93ca-5fe35145f733.xlsx"
    # print(file_downloader.invoke({"url": url, "file_name": file_name}))
