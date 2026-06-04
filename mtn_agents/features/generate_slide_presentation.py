from ..prompt_templates import (
    prompt_generate_slide_bullet_points,
    prompt_generate_slide_description,
    prompt_format_slide_texts,
)
from ..constants import (
    OPENAI_CHAT_MODEL_GPT_4O_MINI,
    JSON_RESPONSE_FORMAT,
    OPENAI_CHAT_MODEL_GPT_4O,
    AUTO_SLIDE_GENERATION_RETRIEVE_THRESHOLD,
)
from ..upload_reference.read_files import read_slides_contents
from ..upload_reference.main import upload_presentation_to_index_pinecone
from ..types import ProductDescriptionDict
from ..utils import Presentation, hex_uuid
from ..retriever import PineconeIndex
from ..llm import OpenAILLM
from ..utils import Slide

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Union, Dict
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def openai_instance():
    return OpenAILLM()


def generate_bullet_points_with_gpt(text: str) -> str:
    variables = {"text": text}

    prompt = prompt_generate_slide_bullet_points.text.format(**variables)

    result = openai_instance().query(
        prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI, response_format=JSON_RESPONSE_FORMAT
    )

    try:
        return json.loads(result)["bullet_points"]
    except json.JSONDecodeError or KeyError as _:
        raise Exception("GPT returned json in a bad format")


def generate_slide_description_with_gpt(slide_text: str, retrieved_text: List[str]) -> str:
    variables = {"slide_text": slide_text, "retrieved_text": "\n\n".join(retrieved_text)}

    prompt = prompt_generate_slide_description.text.format(**variables)

    result = openai_instance().query(
        prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI, response_format=JSON_RESPONSE_FORMAT
    )

    try:
        return json.loads(result)["slide_description"]
    except json.JSONDecodeError or KeyError as _:
        raise Exception("GPT returned json in a bad format")


def format_slides_into_presentation(list_slides: Union[List[str], str]) -> List[Dict[str, str]]:
    variables = {"list_slides": list_slides, "num_slides": len(list_slides)}

    prompt = prompt_format_slide_texts.text.format(**variables)

    result = openai_instance().query(
        prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O, response_format=JSON_RESPONSE_FORMAT
    )

    try:
        return json.loads(result)["output_slides"]
    except json.JSONDecodeError or KeyError as _:
        raise Exception("GPT returned json in a bad format")


def get_data_from_pinecone(text: str, user_id: str, document_ids: List[str], top_k: int) -> List[str]:
    response = PineconeIndex().query(text=text, user_id=user_id, document_ids=document_ids, top_k=top_k)

    return [
        record["metadata"]["text"]
        for record in response["matches"]
        if record["score"] > AUTO_SLIDE_GENERATION_RETRIEVE_THRESHOLD
    ]


def create_slide(slide: Dict[str, str], user_id: str, document_ids: List[str], top_k: int = 10) -> Slide:
    if len(slide["content"]) < 150:
        return Slide(slide_content=slide["content"] + "", slide_number=slide["slide_number"], url=slide["url"])

    list_bullet_points = generate_bullet_points_with_gpt(slide["content"])

    list_retrieved_texts_each_bullet = get_data_from_pinecone(str(list_bullet_points), user_id, document_ids, top_k)

    if list_retrieved_texts_each_bullet:
        slide_additional_generated_description = generate_slide_description_with_gpt(
            slide["content"], list_retrieved_texts_each_bullet
        )

        return Slide(
            slide_content=slide["content"] + slide_additional_generated_description,
            slide_number=slide["slide_number"],
            url=slide["url"],
        )
    else:
        return Slide(slide_content=slide["content"] + "", slide_number=slide["slide_number"], url=slide["url"])


def auto_generate_slide_concurrent(
    slide_path: Union[Path, str, ProductDescriptionDict],
    user_id: str,
    document_ids: List[str],
    agent_uuid: str,
    top_k: int = 10,
) -> ProductDescriptionDict:
    if not isinstance(slide_path, Path):
        slide_path = Path(slide_path)

    if isinstance(slide_path, dict):
        list_slide_texts = Presentation.from_json(slide_path)
    else:
        list_slide_texts = read_slides_contents(path=slide_path)
        if list_slide_texts:
            list_slide_texts = Presentation(list_slide_texts)
        else:
            raise ValueError("The file is empty or could not read it.")

    final_list = [""] * len(list_slide_texts)
    mapping = {hex_uuid()[:6]: item for item in list_slide_texts}

    logger.info(f"\n----------------------\nMappings: {mapping}\n----------------------\n")

    with ThreadPoolExecutor(max_workers=int(8)) as executor:
        future_to_row = {
            executor.submit(create_slide, each_slide[1], user_id, document_ids, top_k): [i]
            for i, each_slide in enumerate(list_slide_texts)
        }

        for future in as_completed(future_to_row):
            idx = future_to_row[future]
            final_list[idx[0]] = future.result()

    logger.info(f"\n----------------------\nList after getting references: {final_list}\n----------------------\n")

    final_list_str = str({key: final_list[idx].content for idx, (key, value) in enumerate(mapping.items())})
    final_list_cleaned = format_slides_into_presentation(final_list_str)

    logger.info(f"\n----------------------\nCleaned list: {final_list_cleaned}\n----------------------\n")
    if len(final_list_cleaned) != len(list_slide_texts) or len(final_list_cleaned) != len(mapping):
        raise IndexError("Length input is not equal to length output in auto slide generation.")

    final_list = []
    for item in final_list_cleaned:
        ((k, v),) = item.items()
        final_list.append(
            Slide(
                slide_content="## slide_content: " + v,
                slide_number=mapping[k][1]["slide_number"],
                url=mapping[k][1]["url"],
            )
        )

    logger.info(f"\n----------------------\nAuto slide generation result: {final_list}\n----------------------\n")

    final_presentation = dict(Presentation(final_list))

    uploaded_data = upload_presentation_to_index_pinecone(
        presentation=final_presentation, user_id=user_id, agent_uuid=agent_uuid
    )

    logger.info(f"\n----------------------\nPresentation in pinecone: {uploaded_data}\n----------------------\n")

    return final_presentation
