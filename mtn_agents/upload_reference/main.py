from typing import Optional, List, Dict, Union

from ..upload_reference.read_files import PrepareData, chunking_and_page_numbering, generate_cache_questions
from ..types import ProductDescriptionDict
from ..retriever import PineconeIndex
from ..utils import add_metadata, Language
from ..constants import LATEST_EMBEDDING_MODEL
from ..types import OpenAITools


def upload_file_to_index_pinecone(
    local_path: str,
    user_id: str,
    document_id: str,
    cache_q: Optional[bool] = False,
    lang: Optional[Language] = Language.SOURCE,
    is_slide: Optional[bool] = False,
    embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
) -> List[Dict[str, Union[str, int]]]:
    """
    Process and upload a file's data to Pinecone Index with optional cache question generation.

    Args:
        - local_path (str): The local path to the file that needs to be uploaded.
        - user_id (str): The user identifier to associate with the uploaded data in Pinecone Index.
        - document_id (str): A unique identifier for the document being uploaded, used as metadata.
        - cache_q (Optional[bool], optional): Flag indicating whether to generate cache questions for
            the document. Defaults to False.
        - lang (Optional[Language], optional): The language of the document, used for cache question
            generation. Defaults to Language.SOURCE.
        - is_slide (Optional[bool], optional): Flag indicating whether the file is a slide. Defaults to False.
        - embedding_model (Optional[str]): The embedding model for using as embedding.

    Returns:
        - List[Dict[str, Union[str, int]]]: The data prepared for Pinecone, after processing,
        metadata addition, and optional cache question generation.
    """
    chunk_type = "document" if not is_slide else "slide"
    prepared_data = PrepareData.prepare_file(file_path=local_path, chunk_type=chunk_type)
    prepared_data = add_metadata(prepared_data, {"document_id": document_id})
    prepared_data = add_metadata(prepared_data, {"embedding_model": embedding_model})

    if cache_q:
        raw_new_prepared_data = generate_cache_questions(prepared_data, lang)
        new_prepared_data = add_metadata(raw_new_prepared_data, {"cache_id": document_id, "document_id": document_id})
        prepared_data += new_prepared_data

    pinecone_instance = PineconeIndex()
    pinecone_data = pinecone_instance.prepare_data_for_pinecone(prepared_data, embedding_model)
    pinecone_instance.detailed_upsert(pinecone_data, user_id)

    return prepared_data


def upload_url_to_index_pinecone(
    url_address: str,
    user_id: str,
    document_id: str,
    cache_q: Optional[bool] = False,
    lang: Optional[Language] = Language.SOURCE,
    embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
    scrape_depth: Optional[int] = 2,
    async_mode: Optional[bool] = False,
) -> List[Dict[str, Union[str, int]]]:
    """
    Processes and uploads an url's data to Pinecone Index with optional cache question generation.

    This function prepares data from an url specified by its url address, adds necessary metadata, and
    optionally generates and includes cache questions based on the file's content. The prepared data,
    now enriched and possibly extended with cache questions, is then uploaded to Pinecone Index under
    a specific user ID.

    Args:
        - url_address (str): The address to the url that needs to be uploaded.
        - user_id (str): The user identifier to associate with the uploaded data in Pinecone Index.
        - document_id (str): A unique identifier for the document being uploaded, used as metadata.
        - cache_q (Optional[bool], optional): Flag indicating whether to generate cache questions for
            the document. Defaults to False.
        - lang (Optional[Language], optional): The language of the document, used for cache question
            generation. Defaults to Language.SOURCE.
        - embedding_model (Optional[str]): The embedding model for using as embedding.
        - scrape_depth (Optional[int]): The depth of scraping recursively sub-pages.

    Returns:
        - List[Dict[str, Union[str, int]]]: The data prepared for Pinecone, after processing,
        metadata addition, and optional cache question generation.
    """
    chunk_type = "url"
    prepared_data = PrepareData.prepare_url(url_address, scrape_depth, chunk_type=chunk_type, async_mode=async_mode)
    prepared_data = add_metadata(prepared_data, {"document_id": document_id})
    prepared_data = add_metadata(prepared_data, {"embedding_model": embedding_model})

    if cache_q:
        raw_new_prepared_data = generate_cache_questions(prepared_data, lang)
        new_prepared_data = add_metadata(raw_new_prepared_data, {"cache_id": document_id, "document_id": document_id})
        prepared_data += new_prepared_data

    pinecone_instance = PineconeIndex()
    pinecone_data = pinecone_instance.prepare_data_for_pinecone(prepared_data, embedding_model=embedding_model)
    pinecone_instance.detailed_upsert(pinecone_data, user_id)

    return prepared_data


def upload_text_to_index_pinecone(
    text: str,
    user_id: str,
    document_id: str,
    cache_q: Optional[bool] = False,
    lang: Optional[Language] = Language.SOURCE,
    embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
) -> List[Dict[str, Union[str, int]]]:
    """
    Processes and uploads a text's data to Pinecone Index with optional cache question generation.

    Args:
        - text (str): The text that needs to be uploaded.
        - user_id (str): The user identifier to associate with the uploaded data in Pinecone Index.
        - document_id (str): A unique identifier for the document being uploaded, used as metadata.
        - cache_q (Optional[bool], optional): Flag indicating whether to generate cache questions for
            the document. Defaults to False.
        - lang (Optional[Language], optional): The language of the document, used for cache question
            generation. Defaults to Language.SOURCE.
        - embedding_model (Optional[str]): The embedding model for using as embedding.

    Returns:
        - List[Dict[str, Union[str, int]]]: The data prepared for Pinecone, after processing,
        metadata addition, and optional cache question generation.
    """
    prepared_data = chunking_and_page_numbering([text], extension="text")
    prepared_data = add_metadata(prepared_data, {"document_id": document_id})
    prepared_data = add_metadata(prepared_data, {"embedding_model": embedding_model})

    if cache_q:
        raw_new_prepared_data = generate_cache_questions(prepared_data, lang)
        new_prepared_data = add_metadata(raw_new_prepared_data, {"cache": document_id})
        prepared_data += new_prepared_data

    pinecone_instance = PineconeIndex()
    pinecone_data = pinecone_instance.prepare_data_for_pinecone(prepared_data, embedding_model=embedding_model)
    pinecone_instance.detailed_upsert(pinecone_data, user_id)

    return prepared_data


def upload_tools_to_index_pinecone(
    tools: OpenAITools,
    tools_description: str,
    user_id: str,
    document_id: str,
    embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
) -> List[Dict[str, Union[str, int]]]:
    """
    Processes and uploads a list of tools' data to Pinecone Index with optional cache question generation.

    Args:
        - tools (List[str]): The list of tools that need to be uploaded.
        - tools_description (str): The description of the tools.
        - user_id (str): The user identifier to associate with the uploaded data in Pinecone Index.
        - document_id (str): A unique identifier for the document being uploaded, used as metadata.
        - embedding_model (Optional[str]): The embedding model for using as embedding.

    Returns:
        - List[Dict[str, Union[str, int]]]: The data prepared for Pinecone, after processing,
        metadata addition, and optional cache question generation.
    """
    prepared_data = chunking_and_page_numbering(list_texts=[tools_description], chunk_type="tools", extension="text")
    prepared_data = add_metadata(prepared_data, {"document_id": document_id})
    prepared_data = add_metadata(prepared_data, {"tools": tools})
    prepared_data = add_metadata(prepared_data, {"embedding_model": embedding_model})

    pinecone_instance = PineconeIndex()
    pinecone_data = pinecone_instance.prepare_data_for_pinecone(prepared_data, embedding_model=embedding_model)
    pinecone_instance.detailed_upsert(pinecone_data, user_id)

    return prepared_data


def upload_presentation_to_index_pinecone(
    presentation: ProductDescriptionDict,
    user_id: str,
    agent_uuid: str,
    embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
) -> List[Dict[str, Union[str, int]]]:
    """
    Processes and uploads a presentation's data to Pinecone Index with optional cache question generation.

    Args:
        - presentation (ProductDescriptionDict): The presentation data that needs to be uploaded.
        - user_id (str): The user identifier to associate with the uploaded data in Pinecone Index.
        - agent_uuid (str): The user identifier to associate with the uploaded data in Pinecone Index.
        - embedding_model (Optional[str]): The embedding model for using as embedding.

    Returns:
        - List[Dict[str, Union[str, int]]]: The data prepared for Pinecone, after processing,
        metadata addition, and optional cache question generation.
    """
    if not presentation:
        raise ValueError("There is no presentation data to upload.")

    prepared_data = []
    for item in presentation.values():
        url = item.get("url", "")
        chunk_type = "url" if url else "slide"
        page_number = (
            int(item.get("slide_number").split("_")[-1])
            if item.get("slide_number").split("_")[-1].isdigit()
            else "None"
        )
        prepared_data.append(
            {
                "source": url if url else "No url",
                "document_id": agent_uuid,
                "embedding_model": embedding_model,
                "text": item.get("content", ""),
                "chunk_type": chunk_type,
                "file_extension": "presentation",
                "page_number": page_number,
            }
        )

    pinecone_instance = PineconeIndex()
    pinecone_data = pinecone_instance.prepare_data_for_pinecone(prepared_data, embedding_model=embedding_model)
    pinecone_instance.detailed_upsert(pinecone_data, user_id=user_id)

    return prepared_data
