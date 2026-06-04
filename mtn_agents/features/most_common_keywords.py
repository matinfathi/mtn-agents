from mtn_agents.utils import convert_string_to_tokens, split_into_chunks
from mtn_agents.constants import JSON_RESPONSE_FORMAT, OPENAI_CHAT_MODEL_GPT_4O_MINI
from mtn_agents.prompt_templates import prompt_most_common_keywords, prompt_most_common_keywords_v2
from mtn_agents.llm import OpenAILLM

from typing import List, Union
import difflib
import time
import json




def find_keywords_v2(target_text: str, words_per_text: Union[str, int] = "10") -> List[str]:
    variables = {"text": target_text, "words_per_text": words_per_text}

    prompt = prompt_most_common_keywords_v2.text.format(**variables)

    keywords_str = OpenAILLM().query(prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI, response_format=JSON_RESPONSE_FORMAT)

    try:
        return json.loads(keywords_str)["words"]
    except json.JSONDecodeError or KeyError as _:
        raise Exception("GPT returned json in a bad format")



def find_keywords(text: str, words_per_text: Union[str, int] = "15") -> List[list]:
    variables = {"text": text, "words_per_text": words_per_text}

    prompt = prompt_most_common_keywords.text.format(**variables)

    keywords_str = OpenAILLM().query(prompt, model=OPENAI_CHAT_MODEL_GPT_4O_MINI, response_format=JSON_RESPONSE_FORMAT)

    try:
        return list(json.loads(keywords_str).values())
    except json.JSONDecodeError or KeyError as _:
        raise Exception("GPT returned json in a bad format")


def find_most_common_keywords_for_english(
    text, max_tokens=4000, words_per_text="1 to 5"
):
    tokens = convert_string_to_tokens(text)

    freq, names = [], []
    for part in split_into_chunks(tokens, max_tokens):
        kws = find_keywords(part, words_per_text)
        time.sleep(2)
        names.extend(kws[0])
        freq.extend(kws[1])

    return [list(set(names)), list(set(freq))]


def is_similar(word1, word2, threshold=0.85):
    similarity_ratio = difflib.SequenceMatcher(None, word1, word2).ratio()
    return similarity_ratio >= threshold


def word_counter(target_text, list_mckw, tag, max_size):
    temp_list = []
    for ii in list(set(list_mckw)):
        word_count = target_text.lower().count(ii.lower())
        if word_count:
            temp_list.append((ii, tag, word_count))
        if len(temp_list) == max_size:
            break

    return sorted(temp_list, key=lambda x: x[2], reverse=True)


def filter_words(selected_words):
    filtered_data = []
    for i, (word1, _, _) in enumerate(selected_words):
        is_unique = all(not is_similar(word1, word2) for word2 in filtered_data)
        if is_unique:
            filtered_data.append(selected_words[i][0])

    return filtered_data


# TO BE REMOVED
def mckw_en(target_text, max_size=10):
    list_mckw = find_most_common_keywords_for_english(target_text)
    selected_name = word_counter(target_text, list_mckw[0], "NAME", max_size)
    selected_freq = word_counter(target_text, list_mckw[1], "FREQ", max_size)

    filtered_name = filter_words(selected_name)
    filtered_freq = filter_words(selected_freq)

    return list(set(filtered_name + filtered_freq))
