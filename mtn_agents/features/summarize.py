from mtn_agents.utils import convert_string_to_tokens, split_into_chunks, Language
from mtn_agents.constants import CONTEXT_WINDOW, SUMMARIZE_LENGTH, OPENAI_CHAT_MODEL_GPT_4O_MINI
from mtn_agents.llm import OpenAILLM

from typing import Union
import time


def summarize_text(
    text: str, summarize_tokens: Union[int, str], lang: Language = Language.SOURCE
):
    prompt = f"""\
**Text:**
```{text}```

**Objective:** Summarize the preceding text within {summarize_tokens} words, \
ensuring all crucial information is included. The summarization language \
should be in {lang}.

**Summary:** """

    result = OpenAILLM().query(
        prompt=prompt,
        model=OPENAI_CHAT_MODEL_GPT_4O_MINI,
    )

    return result


def summarize_text_recursively(
    text: str,
    lang: Language = Language.ENGLISH.description,
):
    tokens = convert_string_to_tokens(text)

    num_tokens = len(tokens)
    if num_tokens <= CONTEXT_WINDOW:
        if num_tokens < SUMMARIZE_LENGTH + 200:
            return text
        return summarize_text(text, SUMMARIZE_LENGTH, lang)

    summaries = []
    for part in split_into_chunks(tokens, CONTEXT_WINDOW):
        summary = summarize_text(part, SUMMARIZE_LENGTH, lang)
        time.sleep(2)
        summaries.append(summary)

    concatenated = " ".join(summaries)
    return summarize_text_recursively(concatenated, CONTEXT_WINDOW)
