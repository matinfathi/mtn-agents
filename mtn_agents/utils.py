from .types import ResponseMetadataDict

from typing import List, Dict, Any, Optional, Union
from functools import lru_cache, wraps
from pathlib import Path
from uuid import uuid4
from enum import Enum
import logging
import pickle
import json
import time
import re

import tiktoken

logger = logging.getLogger(__name__)


class Language(Enum):
    ENGLISH = ("en", "English")
    ARABIC = ("ar", "Arabic")
    SOURCE = ("sr", "language of source text.")

    def __init__(self, code, description):
        self.code = code
        self.description = description


def hex_uuid():
    return uuid4().hex


def get_file_suffix(file_path: Union[str, Path]) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    return file_path.suffix.lower()


def add_metadata(list_: List[Dict[str, Any]], attribute: Dict[str, Any]) -> List[Any]:
    new_list_ = []
    for item in list_:
        item.update(attribute)
        new_list_.append(item)

    return new_list_


@lru_cache
def encoding_for_model(model_name: str = "gpt-3.5-turbo") -> tiktoken.Encoding:
    return tiktoken.encoding_for_model(model_name)


def convert_string_to_tokens(string: str) -> list:
    encoding = encoding_for_model()
    return encoding.encode(string)


def split_into_chunks(tokens, max_tokens):
    num_tokens = len(tokens)

    encoding = encoding_for_model()
    parts = []
    num_parts = num_tokens // max_tokens + 1
    part_size = num_tokens // num_parts

    start = 0
    for _ in range(num_parts):
        end = start + part_size
        part = encoding.decode(tokens[start:end])
        parts.append(part)
        start = end

    return parts


def read_pickle(pickle_path: str) -> Any:
    with open(pickle_path, "rb") as file:
        loaded_item = pickle.load(file)

    return loaded_item


def simple_text_cleaning(text: str) -> str:
    """Clean texts for matching engine."""
    text = re.sub(r"\n\n", r"\n", text)
    text = re.sub(r"\t", r" ", text)
    text = re.sub(r" {2,}", r" ", text)
    text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)
    text = text.replace("\n", " ")
    return text


def cleaning(text: str) -> str:
    """Clean texts for retriever."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([^\w\s])\1+", r"\1", text)
    return text.strip()


def read_txt(path: str) -> str:
    with open(path, "r") as f:
        text = f.read()

    return text


class Reference:
    def __init__(self, score: float, text: str, document_id: Optional[str] = None):
        self.score = score
        self.text = text
        self.document_id = document_id

    def __repr__(self):
        return f"Reference(score={self.score!r}, text={self.text!r}, document_id={self.document_id!r})"


class ResponseMetadata:
    def __init__(
        self,
        intent: str,
        references: List[Reference],
        slide_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt: str = "",
        chain: str = "",
        url: str = "",
        document_id: str = "",
    ):
        self.intent = intent
        self.references = references
        self.slide_id = slide_id
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.prompt = prompt
        self.chain = chain
        self.url = url
        self.document_id = document_id

    def __repr__(self):
        return (
            f"ResponseMetadata(intent={self.intent!r}, slide_id={self.slide_id!r}, references={self.references!r}, "
            f"prompt_tokens={self.prompt_tokens!r}, completion_tokens={self.completion_tokens!r}, "
            f"prompt={self.prompt!r}, chain={self.chain!r}, url={self.url!r}, document_id={self.document_id!r})"
        )

    @property
    def __dict__(self):
        return {
            "intent": self.intent,
            "references": [ref.__dict__ for ref in self.references],  # Assuming Reference has a __dict__ method
            "slide_id": self.slide_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt": self.prompt,
            "chain": self.chain,
            "url": self.url,
            "document_id": self.document_id,
        }

    def __iter__(self):
        yield from self.__dict__.items()


class AgentResponse:
    def __init__(self, data: Dict[str, Any]):
        self.response = data.get("response", "")
        metadata = data.get("metadata", {})
        intent = metadata.get("intent", "")
        slide_id = metadata.get("slide_id", "None")
        references_data = metadata.get("references", [])
        prompt_tokens = metadata.get("prompt_tokens", 0)
        completion_tokens = metadata.get("completion_tokens", 0)
        prompt = metadata.get("prompt", "")
        chain = metadata.get("chain", "")
        url = metadata.get("url", "")
        document_id = metadata.get("document_id", "")
        references = [
            Reference(score=ref_data["score"], text=ref_data["text"], document_id=ref_data.get("document_id"))
            for ref_data in references_data
        ]
        self.metadata = ResponseMetadata(
            intent=intent,
            references=references,
            slide_id=slide_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt=prompt,
            chain=chain,
            url=url,
            document_id=document_id,
        )

    def __repr__(self):
        return f"Response(response={self.response!r}, metadata={self.metadata!r})"

    @property
    def __dict__(self):
        return {"response": self.response, "metadata": self.metadata.__dict__}

    def __iter__(self):
        yield from self.__dict__.items()


def retry_decorator(wait=3, stop=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < stop:
                try:
                    yield from func(*args, **kwargs)
                    break
                except Exception as e:
                    attempts += 1
                    time.sleep(wait)
                    logger.info(f"Retrying after error: {e} and args {args} and kwargs {kwargs}")
                    if attempts == stop:
                        raise Exception("Maximum retry attempts reached") from e

        return wrapper

    return decorator


def encode_visual_sales_agent_answer(answer: str, slide_id: str) -> str:
    return f"""{{"1_used_reference_id": "{slide_id}", "2_text": "{answer}"}}"""


class Dialogue:
    def __init__(
        self,
        user: str,
        assistant: str,
        slide_id: Optional[str] = None,
        assistant_metadata: Optional[ResponseMetadataDict] = None,
    ):
        self.user = user
        self.assistant = assistant
        self.slide_id = slide_id
        self.assistant_metadata = assistant_metadata

    def __repr__(self) -> str:
        return (
            f"Dialogue(user={self.user!r}, assistant={self.assistant!r}, slide_id={self.slide_id!r}, "
            f"assistant_metadata={self.assistant_metadata!r})"
        )


class ChatHistory:
    def __init__(self, chat_history: List[Dialogue]):
        self.chat_history = list(chat_history)
        for item in self.chat_history:
            if type(item) is not Dialogue:
                raise TypeError("Input must be a list of Dialogue objects")
            else:
                if item.slide_id and item.assistant_metadata and item.assistant_metadata.get("chain", "None") == "master_chain":
                    continue
                elif item.assistant_metadata:
                    item.slide_id = item.assistant_metadata.get("slide_id", "None")
                else:
                    item.assistant_metadata = {"slide_id": "None"}
                    item.slide_id = item.assistant_metadata.get("slide_id", "None")

    def __len__(self) -> int:
        return len(self.chat_history)

    def __getitem__(self, item: Union[int, slice]) -> Any:
        if isinstance(item, slice):
            return ChatHistory(
                [
                    Dialogue(
                        user=d.user,
                        assistant=d.assistant,
                        slide_id=d.assistant_metadata.get("slide_id", "None"),
                        assistant_metadata=d.assistant_metadata,
                    )
                    for d in self.chat_history[item]
                ]
            )
        elif isinstance(item, int):
            return self.chat_history[item]
        else:
            raise TypeError("Invalid argument type.")

    def __repr__(self) -> str:
        return f"ChatHistory{self.chat_history!r}"

    def add_first_user_message(self, user_first_message: str):
        self.chat_history = [
            Dialogue(
                user=user_first_message,
                assistant=self.chat_history[0].assistant,
                slide_id=self.chat_history[0].assistant_metadata.get("slide_id", "None"),
                assistant_metadata=self.chat_history[0].assistant_metadata,
            )
        ] + self.chat_history[1:]

    def to_str(
        self,
        assistant_name: Optional[str] = "assistant",
        visual: bool = False,
        chat_history: Optional["ChatHistory"] = None
    ) -> str:
        history = chat_history if chat_history else self.chat_history

        text = ""
        if visual:
            for item in history:
                encoded_answer = encode_visual_sales_agent_answer(answer=item.assistant, slide_id=item.slide_id)
                text += f"- user: {item.user}\n- {assistant_name}: {encoded_answer}\n"
        else:
            for item in history:
                text += f"- user: {item.user}\n- {assistant_name}: {item.assistant}\n"

        return text.strip()



class Slide:
    def __init__(self, slide_number: str, slide_content: str, url: str = None):
        self.slide_number = slide_number
        self.content = slide_content
        self.url = url

    def __repr__(self) -> str:
        return f"Slide(slide_number={self.slide_number!r}, slide_content={self.content!r}, url={self.url})"

    def __iter__(self):
        yield from self.__dict__.items()


def extract_page_number(key: str) -> int:
    return int(key.split("_")[1])


class Presentation:
    def __init__(self, slides: Union[List[str], List[Slide]]):
        temp = {}

        for idx, item in enumerate(slides, start=1):
            if isinstance(item, str):
                slide_number = f"page_{idx}"
                slide = Slide(slide_number=slide_number, slide_content=item)
            elif isinstance(item, Slide):
                slide = item
                slide_number = slide.slide_number
            else:
                raise TypeError("Input must be a list of strings or Slide objects")

            temp[slide_number] = slide

        self._slides = dict(sorted(temp.items(), key=lambda i: int(i[0].split("_")[1])))
        self._slide_order = list(self._slides.keys())

    @classmethod
    def from_json(cls, data: Dict[str, Dict[str, str]]) -> "Presentation":
        slides = []
        for slide_data in data.values():
            slide = Slide(
                slide_number=slide_data["slide_number"],
                slide_content=slide_data["content"],
                url=slide_data.get("url", None),
            )
            slides.append(slide)
        return cls(slides)

    def get_slide(self, slide_number: str) -> Slide:
        return self._slides.get(slide_number, {})

    def __getitem__(self, slide_number: str) -> Slide:
        return self._slides.get(slide_number, {})

    def get_previous_slide(self, slide_number: str) -> Union[Slide, None]:
        try:
            current_index = self._slide_order.index(slide_number)
            if current_index > 0:
                previous_slide_number = self._slide_order[current_index - 1]
                return self._slides[previous_slide_number]
        except ValueError:
            pass
        return None

    def get_next_slide(self, slide_number: str) -> Union[Slide, None]:
        try:
            current_index = self._slide_order.index(slide_number)
            if current_index < len(self._slide_order) - 1:
                next_slide_number = self._slide_order[current_index + 1]
                return self._slides[next_slide_number]
        except ValueError:
            pass
        return None

    def add_slide(self, slide: Slide):
        self._slides[slide.slide_number] = slide
        if slide.slide_number not in self._slide_order:
            self._slide_order.append(slide.slide_number)

    def __setitem__(self, slide_number: str, slide: Slide):
        if slide.slide_number != slide_number:
            raise ValueError("Slide number mismatch")
        self._slides[slide_number] = slide
        if slide_number not in self._slide_order:
            self._slide_order.append(slide_number)

    def __len__(self) -> int:
        return len(self._slides)

    def __iter__(self):
        yield from self.__dict__.items()

    def __repr__(self) -> str:
        return f"Presentation({list(self._slides.values())!r})"

    def to_json(self, to_str: bool = False) -> Union[Dict[str, str], str]:
        final_dict = {item.slide_number: item.content for item in self._slides.values()}
        return json.dumps(final_dict, indent=4) if to_str else final_dict

    def get_url(self, slide_number: str = "page_1"):
        return self._slides[slide_number].url

    @property
    def __dict__(self) -> dict:
        return {slide_number: slide.__dict__ for slide_number, slide in self._slides.items()}

    @property
    def get_final_slide_number(self) -> str:
        if self._slide_order:
            return self._slide_order[-1]
        else:
            raise IndexError("No slides available in the presentation.")


class Prompt:
    def __init__(self, instructions: str, strategy: str, rules: str, example: str):
        self.instructions = instructions
        self.strategy = strategy
        self.rules = rules
        self.example = example

    def __repr__(self) -> str:
        return f"""**Conversation Example (This is an example just mimic the structure):**\n```{self.example}```\n\n"""

    @property
    def get_instructions(self) -> str:
        return self.instructions
