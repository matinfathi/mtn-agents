from mtn_agents.constants import (
    ANTHROPIC_CHAT_MODEL_SONNET_35,
    ANTHROPIC_CHAT_MODEL_MAX_TOKEN,
)
from mtn_agents.utils import retry_decorator
from .base import BaseLLM

from tenacity import retry, stop_after_attempt, wait_fixed
from anthropic import Anthropic

from typing import Optional, Generator
import os


class AnthropicLLM(BaseLLM):
    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        client: Optional[BaseLLM] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("Anthropic API key cannot be 'None'.")

        try:
            self.client = client if client else Anthropic(api_key=api_key)
        except Exception as e:
            raise ValueError(
                f"Anthropic API client failed to initialize. Error: {e}"
            ) from e

    # @retry(wait=wait_fixed(3), stop=(stop_after_attempt(3)))
    def query(
        self,
        prompt: str,
        model: str = ANTHROPIC_CHAT_MODEL_SONNET_35,
        max_tokens: int = ANTHROPIC_CHAT_MODEL_MAX_TOKEN,
        **kwargs,
    ):
        _ = kwargs.pop('response_format', None)
        list_messages = [{"role": "user", "content": prompt}]
        try:
            response = self.client.messages.create(
                messages=list_messages,
                stream=False,
                model=model,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"LLM error: {e}") from e

    # @retry_decorator(wait=3, stop=3)
    def query_stream(
        self,
        prompt: str,
        model: str = ANTHROPIC_CHAT_MODEL_SONNET_35,
        max_tokens: int = ANTHROPIC_CHAT_MODEL_MAX_TOKEN,
        **kwargs,
    ) -> Generator[str, None, None]:
        _ = kwargs.pop('response_format', None)
        list_messages = [{"role": "user", "content": prompt}]
        try:
            stream = self.client.messages.create(
                messages=list_messages,
                model=model,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )

            for event in stream:
                if event.type == "content_block_delta":
                    yield event.delta.text
            return
        except Exception as e:
            raise Exception(f"LLM error: {e}") from e
