from pydantic import BaseModel
from typing import Optional, Dict
import logging

from ..types import OpenAITools, AgentResponseDict
from ..prompt_templates import Prompt
from ..llm import BaseLLM, OpenAILLM, AnthropicLLM
from ..retriever import PineconeIndex
from ..retriever import BaseIndex
from .utils import parse_function_call_stream_call, parse_function_call_stream_no_call


logger = logging.getLogger(__name__)


class BaseChain(BaseModel):
    generator: Optional[BaseLLM] = None
    retriever: Optional[BaseIndex] = None
    model: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        model: Optional[str] = "openai",
        retriever: Optional[BaseIndex] = None,
        generator: Optional[BaseLLM] = None,
        **kwargs,
    ):
        super().__init__(model=model, retriever=retriever, generator=generator, **kwargs)
        if model.lower() == "anthropic":
            self.generator = generator if generator else AnthropicLLM()
        else:
            self.generator = generator if generator else OpenAILLM()
        self.retriever = retriever if retriever else PineconeIndex()

    def __call__(
        self,
        question: Optional[str] = "",
        chat_history_str: Optional[str] = None,
        required_texts: Optional[Dict[str, str]] = None,
        prompt: Optional[Prompt] = "",
        stream: Optional[bool] = False,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[OpenAITools] = None,
    ) -> AgentResponseDict:
        """Processes a question-and-answer (QA) interaction as fast as possible."""
        assembled_prompt = prompt.text.format(chat_history=chat_history_str, question=question, **required_texts)

        response_format = response_format if response_format else {"type": "text"}

        logger.info(f"\n----------------------\nUsing Base Chain --> {assembled_prompt}\n----------------------\n")

        list_messages = [{"role": "user", "content": assembled_prompt}]

        response = self.generator.detailed_query(
            list_messages=list_messages, stream_response=stream, tools=tools, response_format=response_format
        )

        if stream:
            item1 = next(response)
            if item1.choices[0].delta.content is None:
                function_call = parse_function_call_stream_call(response, item1)
                final_response = None
                response_type = "function_call"
            else:
                function_call = None
                final_response = parse_function_call_stream_no_call(response, item1)
                response_type = "response"
        else:
            if response.choices[0].message.tool_calls is not None:
                function_call = response.choices[0].message.tool_calls
                response_type = "function_call"
            else:
                function_call = None
                response_type = "response"

            final_response = response.choices[0].message.content

        final_response = {
            "response": final_response,
            "response_type": response_type,
            "metadata": {
                "prompt_tokens": self.generator.prompt_tokens,
                "completion_tokens": self.generator.completion_tokens,
                "prompt": assembled_prompt,
                "function_call": function_call,
            },
        }

        return final_response
