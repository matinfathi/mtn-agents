from typing import Generator, Optional, Union, List, Dict


def stream_sentence_with_switch(
    word_by_word_generator: Generator[str, None, None],
    end_sentence_sign: Optional[str] = None,
    clean_each_sentence: Optional[bool] = False,
    idk_list: Optional[List[str]] = None,
) -> Generator[Union[str, bool], None, None]:
    current_sentence = ""
    number_of_sentences = 0
    if not end_sentence_sign:
        end_sentence_list = [".", "!", "?", ":", "\n"]
    else:
        end_sentence_list = [end_sentence_sign]

    for token in word_by_word_generator:
        current_sentence += token if type(token) is str else ""
        if any(True if current_sentence.strip().endswith(item) else False for item in end_sentence_list):
            final_sentence = current_sentence.split(end_sentence_sign)[0] if clean_each_sentence else current_sentence
            if idk_list and any(
                (
                    True
                    if final_sentence.lower()
                    == (idk_str.split(end_sentence_sign)[0].lower() if clean_each_sentence else idk_str.lower())
                    else False
                )
                for idk_str in idk_list
            ):
                yield False
                return None

            if not number_of_sentences:
                yield True
            yield final_sentence
            current_sentence = ""
            number_of_sentences += 1

    if not number_of_sentences:
        yield True
    yield current_sentence


def convert_information_to_prompt(info_dict: Dict[str, str]) -> str:
    output = []

    for key, value in info_dict.items():
        output.append(f"**{key.capitalize().replace('_', ' ')}:**\n{value}\n")

    output_text = "\n".join(output)

    return output_text


def encode_visual_sales_agent_answer(answer: str, slide_id: str) -> str:
    return f"""{{"1_used_reference_id": "{slide_id}", "2_text": "{answer}"}}"""
