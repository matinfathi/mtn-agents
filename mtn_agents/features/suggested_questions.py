from ..utils import convert_string_to_tokens, split_into_chunks, Language, ChatHistory, Presentation
from ..constants import OPENAI_CHAT_MODEL_GPT_4O_MINI, JSON_RESPONSE_FORMAT
from ..types import ResponseMetadataDict, ProductDescriptionDict, FilterDict
from ..prompt_templates import prompt_suggested_question
from ..retriever import PineconeIndex
from ..llm import OpenAILLM

from typing import List, Dict, Union, Optional
import json
import time
import logging

logger = logging.getLogger(__name__)


def prepare_filters_for_cache(document_ids: List[str]) -> FilterDict:
    filters_for_cache = {
        "document_id": {"$in": document_ids},
        "cache_id": {"$in": document_ids},
    }

    return filters_for_cache


def prepare_filters_for_text(document_ids: List[str]) -> FilterDict:
    filters_for_cache = {
        "document_id": {"$in": document_ids},
    }

    return filters_for_cache


def generate_suggested_questions_with_answer(
    text: str,
    num_question: Union[str, int],
    language: Optional[Language] = Language.SOURCE,
) -> List[Dict[str, str]]:
    variables = {
        "text": text,
        "num_question": num_question,
        "language": language.description,
    }
    prompt = prompt_suggested_question.text.format(**variables)

    result: str = OpenAILLM().query(
        prompt=prompt,
        model=OPENAI_CHAT_MODEL_GPT_4O_MINI,
        response_format=JSON_RESPONSE_FORMAT,
    )
    result: Dict[str, Dict[str, str]] = json.loads(result)

    return [
        {"question": item["question"], "answer": item["answer"]}
        for item in result.values()
    ]


def generate_suggested_questions_with_answer_v2(
    text: str,
    chat_history: ChatHistory,
    num_question: Union[str, int],
    language: Optional[Language] = Language.SOURCE,
) -> List[Dict[str, str]]:
    variables = {
        "text": text,
        "num_question": num_question,
        "language": language.description,
        "chat_history": chat_history.to_str(),
    }
    prompt = prompt_suggested_question.text.format(**variables)

    result: str = OpenAILLM().query(
        prompt=prompt,
        model=OPENAI_CHAT_MODEL_GPT_4O_MINI,
        response_format=JSON_RESPONSE_FORMAT,
    )
    result: Dict[str, Dict[str, str]] = json.loads(result)

    return [
        {"question": item["question"], "answer": item["answer"]}
        for item in result.values()
    ]


def suggest_questions(
    text, num_question="3 to 5", lang: Language = Language.ENGLISH
) -> List[Dict[str, str]]:
    tokens = convert_string_to_tokens(text)

    data_list = []
    for part in split_into_chunks(tokens, 4000):
        chunk_list_qa = generate_suggested_questions_with_answer(
            part, num_question, lang.description
        )
        data_list.extend(chunk_list_qa)

        # Sleep for 2 seconds to avoid rate limit
        # TODO remove if it's working
        time.sleep(2)

    return data_list


def get_suggested_question_from_retriever(
    text: str,
    top_k: int,
    user_id: str,
    document_ids: List[str],
    similarity: Optional[float] = 0.83,
) -> List[Union[Dict[str, str], None]]:
    response_list = []

    filters = prepare_filters_for_cache(document_ids)

    pc = PineconeIndex()
    response = pc.detailed_query(
        input_texts=[text],
        user_id=user_id,
        filter_dict=filters,
        top_k=top_k,
    )

    if not len(response.get("matches", [])):
        return []

    for item in response.get("matches", []):
        if item["score"] >= similarity:
            try:
                response_list.append(json.loads(item["metadata"]["text"]))
            except json.JSONDecodeError or ValueError:
                continue

    return response_list


def get_text_from_retriever(
    text: str,
    top_k: int,
    user_id: str,
    document_ids: List[str],
    similarity: Optional[float] = 0,
) -> str:
    response_list = []

    filters = prepare_filters_for_text(document_ids)

    pc = PineconeIndex()
    response = pc.detailed_query(
        input_texts=[text],
        user_id=user_id,
        filter_dict=filters,
        top_k=top_k,
    )

    if not len(response.get("matches", [])):
        return ""

    for item in response.get("matches", []):
        if item["score"] >= similarity:
            try:
                response_list.append(item["metadata"]["text"])
            except json.JSONDecodeError or ValueError:
                continue

    return "\n\n".join(response_list)


def get_followup_question(
    text: Union[ChatHistory, str],
    user_id: Optional[str] = None,
    document_ids: Optional[List[str]] = None,
    top_k: Optional[int] = 3,
    cache: Optional[bool] = False,
    language: Optional[Language] = Language.ENGLISH,
) -> List[Dict[str, str]]:
    cache_q, generated_q = [], []
    if type(text) is ChatHistory:
        text = text.to_str()

    if cache:
        if not text:
            text = "abc"
            cache_q = get_suggested_question_from_retriever(
                text, top_k, user_id, document_ids, similarity=0
            )
        else:
            cache_q = get_suggested_question_from_retriever(
                text, top_k, user_id, document_ids
            )

    generate_num = top_k - len(cache_q)

    if generate_num > 0:
        if text:
            generated_q = generate_suggested_questions_with_answer(
                text=text, num_question=generate_num, language=language
            )
        else:
            text2 = get_text_from_retriever(text="abc", user_id=user_id, document_ids=document_ids, top_k=top_k)
            generated_q = generate_suggested_questions_with_answer(
                text=text2, num_question=generate_num, language=language
            )

    return_answer = cache_q + generated_q

    return (
        return_answer
        if len(return_answer) == top_k
        else return_answer[: max(len(return_answer), top_k)]
    )


def get_followup_question_v2(
    product_description: ProductDescriptionDict,
    chat_history: ChatHistory,
    response_metadata: ResponseMetadataDict,
    top_k: Optional[int] = 3,
    language: Optional[Language] = Language.ENGLISH,
) -> List[Dict[str, str]]:
    product_description = Presentation.from_json(product_description)
    json_pd = product_description.to_json()
    str_pd = product_description.to_json(to_str=True)
    logger.info(f"\n----------------------\nProduct in suggested questions v2 --> {str_pd}\n----------------------\n")
    if chat_history:
        current_slide_id = chat_history[-1].slide_id
        reference_text = ""
        if current_slide_id != "None":
            current_slide_text = json_pd.get(current_slide_id, str_pd)
            prefix, num_str = current_slide_id.split("_")
            next_slide_text = json_pd.get(f"{prefix}_{int(num_str)+1}", "")
            if "references" in response_metadata.keys():
                current_slide_text = json_pd.get(current_slide_id, str_pd)
                reference_text = "\n".join(
                    [
                        item["text"]
                        for item in response_metadata["references"][
                                    : max(len(response_metadata["references"]), top_k)
                                    ]
                    ]
                )
            target_text = current_slide_text + "\n\n" + next_slide_text + "\n\n" + reference_text
        else:
            target_text = str_pd
    else:
        target_text = str_pd

    logger.info(f"\n----------------------\nTarget text in suggested questions v2 --> {target_text}\n----------------------\n")

    generated_q = generate_suggested_questions_with_answer_v2(
        text=target_text,
        chat_history=chat_history,
        num_question=top_k,
        language=language,
    )

    logger.info(
        f"\n----------------------\nsuggested questions v2 --> {generated_q}\n----------------------\n")

    return (
        generated_q
        if len(generated_q) == top_k
        else generated_q[: max(len(generated_q), top_k)]
    )
