from mtn_agents.constants import OPENAI_CHAT_MODEL_GPT_4O_MINI
from mtn_agents.llm.base import BaseLLM
from mtn_agents.prompt_templates import prompt_standalone_question
from partialjson.json_parser import JSONParser

from typing import List, TypedDict, Dict, Union, Optional, Generator, Any
import json


def create_standalone_question(llm: BaseLLM, chat_history: str, question: str) -> str:
    """Create standalone question based on chat history and final question."""
    new_prompt = prompt_standalone_question.text.format(question=question, chat_history=chat_history)
    response = llm.query(prompt=new_prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI)

    return response


def process_response(reference: str) -> str:
    """Process the response."""
    try:
        response_dict = json.loads(reference)
    except json.JSONDecodeError:
        return ""

    answer = response_dict.get("response", response_dict.get("answer", response_dict))
    return answer


class FilterDict(TypedDict):
    document_id: Dict[str, List[str]]
    embedding_model: Dict[str, List[str]]
    cache_id: Optional[Dict[str, Union[List[str], bool]]]


def prepare_filters_for_cache(
    document_ids: List[str], cache_list: List[str], embedding_model: str
) -> FilterDict:
    concat_list = document_ids + cache_list
    filters_for_cache = {
        "document_id": {"$in": concat_list},
        "embedding_model": {"$in": [embedding_model]},
        "cache_id": {"$in": concat_list},
    }

    return filters_for_cache


def prepare_filters_for_reference(
    document_ids: List[str],
    embedding_model: str,
    agent_uuid: str,
    main_document_id: str,
) -> FilterDict:
    new_document_ids = list(document_ids)
    if agent_uuid and main_document_id:
        new_document_ids.append(agent_uuid)
        new_document_ids.remove(main_document_id)
    else:
        new_document_ids = document_ids
    filters_for_ref = {
        "document_id": {"$in": new_document_ids},
        "embedding_model": {"$in": [embedding_model]},
        "cache_id": {"$exists": False},
    }

    return filters_for_ref


def decode_visual_sales_agent_answer_stream(visual_sales_agent_answer: Generator[str, None, None]) -> Generator[str, None, None]:
    def init_parser():
        return JSONParser()

    parser = init_parser()

    full_string = ""
    first_token = True
    full_answer_text = ""

    for token in visual_sales_agent_answer:
        full_string += token if token is not None else ""
        if full_string.strip():
            answer_dict = parser.parse(full_string)
            if "1_used_reference_id" in list(answer_dict.keys()) and "2_text" in list(answer_dict.keys()):
                if first_token:
                    yield answer_dict.get("1_used_reference_id", "None")
                    first_token = False
                else:
                    if list(answer_dict.keys())[0] == "2_text":
                        yield answer_dict.get("2_text", "")
                        return
                    else:
                        answer = answer_dict.get("2_text", "")
                        if answer:
                            yield answer[len(full_answer_text):]
                            full_answer_text = answer

    if not full_string:
        yield ""


def parse_function_call_stream_no_call(response: Any, item1: Any) -> Generator[Any, None, None]:
    yield item1.choices[0].delta.content
    for item in response:
        yield item.choices[0].delta.content


def parse_function_call_stream_call(response: Any, item1: Any) -> List[Dict[str, str]]:
    function_call = []
    text_args = ""
    if item1.choices[0].delta.tool_calls is None:
        item1 = next(response)
    
    text_func_name = item1.choices[0].delta.tool_calls[0].function.name
    
    for item in response:
        tool_call = item.choices[0].delta.tool_calls[0] if item.choices[0].delta.tool_calls else None
        
        if tool_call and tool_call.function.name is None:
            text_args += tool_call.function.arguments if tool_call.function.arguments else ""
        elif tool_call and tool_call.function.name is not None:
            if text_func_name and text_args:
                function_call.append({"function": text_func_name, "arguments": text_args})
            text_args = ""
            text_func_name = tool_call.function.name
        else:
            if text_func_name and text_args:
                function_call.append({"function": text_func_name, "arguments": text_args})
                text_func_name = ""
                text_args = ""

    if text_func_name and text_args:
        function_call.append({"function": text_func_name, "arguments": text_args})
    
    return function_call
