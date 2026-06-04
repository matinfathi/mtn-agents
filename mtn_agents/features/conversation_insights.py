from mtn_agents.prompt_templates import prompt_conversation_insights
from mtn_agents.constants import OPENAI_CHAT_MODEL_GPT_4O
from mtn_agents.utils import ChatHistory
from mtn_agents.llm import OpenAILLM

from typing import Dict, Any
import json


def get_conv_insights(
        chat_history: ChatHistory
) -> Dict[str, Any]:
    variables = {
        "chat_history": chat_history.to_str(assistant_name="AI Agent")
    }

    prompt = prompt_conversation_insights.text.format(**variables)

    result = OpenAILLM().query(
        prompt=prompt, model=OPENAI_CHAT_MODEL_GPT_4O, response_format={"type": "json_object"}
    )

    try:
        return json.loads(result)
    except json.JSONDecodeError as _:
        raise Exception("GPT returned json in a bad format")
