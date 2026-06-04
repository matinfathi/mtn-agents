from ..constants import OPENAI_CHAT_MODEL_GPT_4O, LATEST_EMBEDDING_MODEL, OPENAI_REASONING_MODEL_O1
from ..types import OpenAITools
from ..utils import retry_decorator
from .base import BaseLLM

from openai.types.chat.chat_completion import ChatCompletion
from tenacity import retry, stop_after_attempt, wait_fixed
from openai._streaming import Stream
from openai import OpenAI, AsyncOpenAI

from typing import Optional, List, Generator, Iterable, Dict
import os


class OpenAILLM(BaseLLM):
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        client: Optional[BaseLLM] = None,
        async_client: Optional[BaseLLM] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key cannot be 'None'.")

        try:
            self.client = client if client else OpenAI(api_key=api_key)
            self.async_client = async_client if async_client else AsyncOpenAI(api_key=api_key)
        except Exception as e:
            raise ValueError(f"OpenAI API client failed to initialize. Error: {e}") from e

    def detailed_query(
        self,
        list_messages: List[Dict[str, str]],
        stream_response: bool = False,
        model: str = OPENAI_CHAT_MODEL_GPT_4O,
        tools: Iterable[OpenAITools] = None,
        **kwargs,
    ) -> ChatCompletion | Stream[ChatCompletion]:
        try:
            response = self.client.chat.completions.create(
                messages=list_messages, stream=stream_response, model=model, tools=tools, **kwargs
            )
            return response
        except Exception as e:
            raise Exception(f"LLM error: {e}") from e

    @retry(wait=wait_fixed(3), stop=(stop_after_attempt(3)))
    def query(self, prompt: str, model: str = OPENAI_CHAT_MODEL_GPT_4O, **kwargs) -> str:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        list_messages = [{"role": "user", "content": prompt}]
        response = self.detailed_query(list_messages=list_messages, model=model, stream_response=False, **kwargs)
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    @retry_decorator(wait=3, stop=3)
    def query_stream(self, prompt: str, model: str = OPENAI_CHAT_MODEL_GPT_4O, **kwargs) -> Generator[str, None, None]:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        list_messages = [{"role": "user", "content": prompt}]
        response = self.detailed_query(
            list_messages=list_messages,
            model=model,
            stream_response=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        for token in response:
            if token.choices and token.choices[0].delta.content is not None:
                yield token.choices[0].delta.content
            if token.usage is not None:
                self.prompt_tokens += token.usage.prompt_tokens
                self.completion_tokens += token.usage.completion_tokens
        return

    def reasoning(self, prompt: str, model: str = OPENAI_REASONING_MODEL_O1, **kwargs) -> str:
        list_messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            messages=list_messages, model=model, **kwargs
        )

        return response.choices[0].message.content

    @retry(wait=wait_fixed(3), stop=(stop_after_attempt(3)))
    def embedding(self, list_texts: List[str], model: str = LATEST_EMBEDDING_MODEL, **kwargs) -> List[List[float]]:
        for idx, item in enumerate(list_texts):
            if not item:
                raise Exception(f"Data contain empty values. List has {len(list_texts)}, and item {idx + 1} is empty.")
        try:
            response = self.client.embeddings.create(input=list_texts, model=model, **kwargs)

            return [d.embedding for d in response.data]
        except Exception as e:
            raise Exception(f"Embedding error: {e}") from e


def query_stream_sentence(
    text: Generator[str, None, None], end_sign: Optional[str] = None, trim: Optional[bool] = False
) -> Generator[str, None, None]:
    sentence = ""
    if not end_sign:
        end_sentence_list = [".", "!", "?", ":", "\n"]
    else:
        end_sentence_list = [end_sign]

    for token in text:
        sentence += token if type(token) is str else ""
        if any([True if sentence.strip().endswith(item) else False for item in end_sentence_list]):
            yield sentence.split(end_sign)[0] if trim else sentence
            sentence = ""

    yield sentence
