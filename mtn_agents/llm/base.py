from pydantic import BaseModel

from typing import Optional, Any


class BaseLLM(BaseModel):
    client: Optional[Any] = None
    async_client: Optional[Any] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def query(self, prompt: str, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
