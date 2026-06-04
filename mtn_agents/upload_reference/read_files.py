from typing import List, Union, Dict
from pathlib import Path
import json

from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import CharacterTextSplitter
from semantic_router.encoders import OpenAIEncoder
from semantic_chunkers import StatisticalChunker
from bs4 import BeautifulSoup as Soup
from pptx import Presentation
import pandas as pd
import docx2txt
import fitz

from ..utils import Language, simple_text_cleaning
from ..constants import LATEST_EMBEDDING_MODEL


def chunking_by_character(text: str, **kwargs) -> List[str]:
    """Chunk and split a text by number of characters."""
    try:
        text_splitter = CharacterTextSplitter(
            separator=kwargs.get("separator", ""),
            chunk_size=kwargs.get("chunk_size", 1000),
            chunk_overlap=kwargs.get("chunk_overlap", 100),
            length_function=kwargs.get("length_function", len),
            is_separator_regex=kwargs.get("is_separator_regex", False),
        )

        texts = text_splitter.create_documents([text])

    except Exception as e:
        raise Exception(f"Text splitting failed. Error {e}") from e

    texts = [d.page_content for d in texts]

    if len(texts) >= 2 and len(texts[-1]) < kwargs.get("chunk_size", 1000) / 2:
        texts[-2] += " " + texts[-1]
        texts.pop()

    return texts


def chunking_by_semantic(text: str, **kwargs) -> List[str]:
    encoder = OpenAIEncoder(name=LATEST_EMBEDDING_MODEL)
    chunker = StatisticalChunker(encoder=encoder)

    chunks = chunker(docs=[text])

    return [" ".join(chunk.splits) for chunk in chunks[0]]


def chunking_and_page_numbering(
    list_texts: List[str], extension: str, chunk_type: str = "document"
) -> List[Dict[str, Union[str, int]]]:
    """Chunk texts and add page number"""
    chunks = []
    for idx, text in enumerate(list_texts):
        if chunk_type == "slide":
            page_content_chunks = [text]
        else:
            page_content_chunks = chunking_by_semantic(text=text)

        for page_content_chunk in page_content_chunks:
            if extension == "pdf":
                chunks.append(
                    {
                        "content": page_content_chunk,
                        "page_number": idx + 1,
                        "file_extension": extension,
                        "chunk_type": chunk_type,
                    }
                )
            else:
                chunks.append({"content": page_content_chunk, "file_extension": extension, "chunk_type": chunk_type})

    return chunks


def chunking_and_source(list_texts: List[Dict[str, str]]) -> List[Dict[str, Union[str, int]]]:
    chunks = []

    for item in list_texts:
        page_content_chunks = chunking_by_semantic(text=item["content"])
        for page_content_chunk in page_content_chunks:
            chunks.append(
                {"content": page_content_chunk, "source": item["source"], "chunk_type": "url", "file_extension": "url"}
            )

    return chunks


def read_pdf(path: Path) -> List[str]:
    """Read the content of PDF files."""
    if not path:
        raise ValueError("There is no file to read.")

    pdf_content = []
    try:
        reader = fitz.open(filename=path, filetype="pdf")
    except Exception as e:
        raise Exception(f"Failed to read PDF. Error {e}") from e
    for page in reader:
        page_content = simple_text_cleaning(page.get_text()).strip()
        pdf_content.append(page_content)

    return pdf_content


def read_docx(path: Path) -> List[str]:
    """Read the content of DOCX files."""
    if not path:
        raise ValueError("There is no file to read.")

    try:
        with open(path, "rb") as f:
            doc = docx2txt.process(f)
    except Exception as e:
        raise Exception(f"Failed to read DOCX. Error {e}") from e

    content = simple_text_cleaning(doc)

    return [content]


def read_txt(path: Path) -> List[str]:
    """Read the content of TXT files."""
    if not path:
        raise ValueError("There is no file to read.")

    try:
        with path.open(mode="r", encoding="utf-8") as f:
            doc = f.read()
    except Exception as e:
        raise Exception(f"Failed to read TXT. Error {e}") from e

    content = simple_text_cleaning(doc)

    return [content]


def read_pptx_contents(path: Path) -> List[str]:
    if not path:
        raise ValueError("There is no file to read.")

    contents = []
    try:
        prs = Presentation(path)
    except Exception as e:
        raise Exception(f"Failed to read PPTX. Error {e}") from e
    for slide in prs.slides:
        slides_texts = ""
        slide_note = slide.notes_slide.notes_text_frame.text
        if slide_note.startswith('"') and slide_note.endswith('"'):
            slide_note = slide_note[1:-1]
        slides_texts += slide_note + "\n"
        for shapes in slide.shapes:
            if shapes.has_text_frame:
                text = shapes.text
                if text:
                    slides_texts += text + " "

        contents.append(simple_text_cleaning(slides_texts.strip()))

    return contents


def read_pdf_and_chunking(path: Path, chunk_type: str = "document") -> List[Dict[str, Union[str, int]]]:
    """Read the content of PDF files and chunk it."""
    list_texts = read_pdf(path)
    return chunking_and_page_numbering(list_texts, extension="pdf", chunk_type=chunk_type)


def read_docx_and_chunking(path: Path, chunk_type: str, **kwargs) -> List[Dict[str, Union[str, int]]]:
    """Read the content of DOCX files and chunk it."""
    list_texts = read_docx(path)
    return chunking_and_page_numbering(list_texts, extension="docx")


def read_txt_and_chunking(path: Path, chunk_type: str) -> List[Dict[str, Union[str, int]]]:
    """Read the content of TXT files and chunk it."""
    list_texts = read_txt(path)
    return chunking_and_page_numbering(list_texts, extension="txt")


def read_xlsx_and_chunking(path: Path, chunk_type: str) -> List[Dict[str, Union[str, int]]]:
    """Read the content of XLSX files."""
    if not path:
        raise ValueError("There is no file to read.")

    contents = []
    try:
        excel = pd.read_excel(path)
    except Exception as e:
        raise Exception(f"Failed to read PDF. Error {e}") from e
    for row in excel.itertuples():
        text = " | ".join([str(item) for item in row])
        text = simple_text_cleaning(text)
        contents.append({"content": text, "file_extension": "xlsx"})

    return contents


def read_pptx_and_chunking(path: Path, chunk_type: str = "slide") -> List[Dict[str, Union[str, int]]]:
    """Read the content of PPTX files."""
    if not path:
        raise ValueError("There is no file to read.")

    contents = []
    try:
        prs = Presentation(path)
    except Exception as e:
        raise Exception(f"Failed to read PPTX. Error {e}") from e
    for idx, slide in enumerate(prs.slides):
        slides_texts = ""
        slide_note = slide.notes_slide.notes_text_frame.text
        if slide_note.startswith('"') and slide_note.endswith('"'):
            slide_note = slide_note[1:-1]
        slides_texts += slide_note + "\n"
        for shapes in slide.shapes:
            if shapes.has_text_frame:
                text = shapes.text
                if text:
                    slides_texts += text + " "
        text = simple_text_cleaning(slides_texts)
        contents.append({"content": text, "file_extension": "pptx", "page_number": idx + 1, "chunk_type": chunk_type})

    return contents


def read_slides_contents(path: Path) -> List[str]:
    extension_to_function = {".pdf": read_pdf, ".pptx": read_pptx_contents}

    func = extension_to_function.get(path.suffix.lower())

    return func(path)


def scrape(url: str, scrape_depth: int = 2, async_mode: bool = True) -> List[Dict[str, str]]:
    unwanted_extensions = (".woff2", ".jpg", ".png", ".gif", ".svg", ".js", ".css", ".ttf", ".otf", ".eot")
    scrape_list = []
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=scrape_depth,
        extractor=lambda x: Soup(x, "html.parser").text,
        use_async=async_mode,
        prevent_outside=True,
    )
    docs = loader.load()
    if (not docs) or (len(docs) <= 1 and len(docs[0].page_content) < 500):
        raise ValueError(
            "We were unable to retrieve the content from the URL you provided. This could be due to "
            "the website being empty or policies preventing content extraction."
        )
    for doc in docs:
        if any(doc.metadata["source"].endswith(ext) for ext in unwanted_extensions):
            continue
        txt = simple_text_cleaning(doc.page_content) if doc else ""
        scrape_list.append({"source": doc.metadata["source"], "content": txt})

    return scrape_list


class PrepareData:
    @staticmethod
    def prepare_file(file_path: str, chunk_type: str) -> List[Dict[str, Union[str, int]]]:
        """Accept a file path and base on the extension of the file, read it and return data."""
        extension_to_function = {
            ".pdf": read_pdf_and_chunking,
            ".wpdf": read_pdf_and_chunking,
            ".docx": read_docx_and_chunking,
            ".txt": read_txt_and_chunking,
            ".xlsx": read_xlsx_and_chunking,
            ".pptx": read_pptx_and_chunking,
        }

        if not isinstance(file_path, Path):
            file_path = Path(file_path)

        func = extension_to_function.get(file_path.suffix.lower())

        if func:
            result = func(file_path, chunk_type)
            return result
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    @staticmethod
    def prepare_url(
        url_address: str, scrape_depth: int, chunk_type: str, async_mode: bool
    ) -> List[Dict[str, Union[str, int]]]:
        list_texts = scrape(url_address, scrape_depth, async_mode)
        return chunking_and_source(list_texts)

    @staticmethod
    def prepare_text(text: str, extension: str, chunk_type: str) -> List[Dict[str, Union[str, int]]]:
        """Accepts a text and chunk it."""
        return chunking_and_page_numbering([text], extension, chunk_type)


def generate_cache_questions(
    data: List[Dict[str, Union[str, int]]], language: Language
) -> List[Dict[str, Union[str, int]]]:
    new_data = []
    for item in data:
        _ = item.get("content", "")
        page_number = item.get("page_number", None)
        extension = item.get("file_extension", "")
        # raw_cache_question_answers = generate_suggested_questions_with_answer(
        #     text, num_question=2, language=language
        # )
        raw_cache_question_answers = []
        for each_cache_q in raw_cache_question_answers:
            if page_number is None:
                new_data.append({"content": json.dumps(each_cache_q), "file_extension": extension})
            else:
                new_data.append({"content": str(each_cache_q), "page_number": page_number, "file_extension": extension})

    return new_data
