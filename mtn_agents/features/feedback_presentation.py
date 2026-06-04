from mtn_agents.constants import OPENAI_CHAT_MODEL_GPT_4O_MINI, JSON_RESPONSE_FORMAT
from mtn_agents.prompt_templates import prompt_slide_feedback
from mtn_agents.utils import Presentation
from mtn_agents.llm import OpenAILLM

from functools import lru_cache
import json


@lru_cache(maxsize=1)
def openai_instance():
    return OpenAILLM()


def get_presentation_feedback(product_description: Presentation) -> str:
    llm = openai_instance()

    prompt = prompt_slide_feedback.text.format(presentation=product_description.to_json(to_str=True))
    answer = llm.query(prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI, response_format=JSON_RESPONSE_FORMAT)

    return json.loads(answer).get("feedback", "")
