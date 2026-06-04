from mtn_agents.prompt_templates import prompt_visual_sales_agent_v36, prompt_call_to_action
from mtn_agents.constants import OPENAI_CHAT_MODEL_GPT_4O_MINI, JSON_RESPONSE_FORMAT
from mtn_agents.llm import OpenAILLM

from partialjson.json_parser import JSONParser

from functools import lru_cache
import asyncio

queue_cta = []
queue_qa = []
call_to_action_switch = None


@lru_cache(maxsize=1)
def init_():
    return JSONParser(), OpenAILLM()


parser, llm = init_()


def decide_call_to_action_switch():
    global call_to_action_switch
    full_text = "".join([item for item in queue_cta])
    answer_dict = parser.parse(full_text)
    if "call_to_action" in list(answer_dict.keys()) and "response" in list(answer_dict.keys()):
        call_to_action_switch = answer_dict.get("call_to_action", False)
    else:
        call_to_action_switch = None


async def cta_request(prompt):
    global call_to_action_switch
    req = llm.aquery_stream(prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI,
                            response_format=JSON_RESPONSE_FORMAT)
    async for token in req:
        if token is not None:
            queue_cta.append(token)
            decide_call_to_action_switch()
            if call_to_action_switch is None:
                continue
            elif call_to_action_switch is True:
                queue_cta.append(req)
                break
            else:
                break


async def qa_request(prompt):
    global call_to_action_switch
    req = llm.aquery_stream(prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI,
                            response_format=JSON_RESPONSE_FORMAT)
    async for token in req:
        queue_qa.append(token)
        if call_to_action_switch is False:
            queue_qa.append(req)
            break
        elif call_to_action_switch is True:
            break
        else:
            continue


def iter_over_async(it, ait, loop):
    for item in it:
        yield item

    ait = ait.__aiter__()

    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj


def concurrent_requests(question, chat_history_str, required_texts):
    prompt_cta = prompt_call_to_action.text.format(question=question, chat_history=chat_history_str)
    prompt_qa = prompt_visual_sales_agent_v36.text.format(question=question, chat_history=chat_history_str, **required_texts)
    loop = asyncio.get_event_loop()

    async def concurrent_requests():
        tasks = [
            asyncio.create_task(cta_request(prompt=prompt_cta)),
            asyncio.create_task(qa_request(prompt=prompt_qa))
        ]
        await asyncio.gather(*tasks)

    loop.run_until_complete(concurrent_requests())

    if call_to_action_switch:
        sync_list = []
        async_gen = queue_cta[-1]
    else:
        sync_list = queue_qa[:-1]
        async_gen = queue_qa[-1]

    final_unified_generator = iter_over_async(sync_list, async_gen, loop)

    return final_unified_generator
