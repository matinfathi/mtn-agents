from mtn_agents.agents.sales_agent import VisualSalesAgent
from mtn_agents.constants import JSON_RESPONSE_FORMAT
from mtn_agents.utils import Presentation, ChatHistory, Dialogue
from mtn_agents.llm import OpenAILLM
from my_tests import my_test_base

import pandas as pd
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, Dict, List, Tuple
import pathlib
import json
import time
import re
import os

llm = OpenAILLM()
vsa = VisualSalesAgent(use_master_chain=False)

document_id_list = [
    "4055933e",
    "f128b160",
    "c6318e57",
    "f79f179d",
    "a75bf192",
    "3f24c5fa",
    "ca9efe14",
    "0a21192a",
    "d173cdc8",
    "87d4b063",
    "b1dbf309",
    "682b40ed",
    "255489a1",
    "b8805df4",
    "f9a6c0d8",
    "302b03ce",
    "7cf59609",
    "abad8baf",
    "y7tef234",
    "aws",
    "coursera",
    "nexus",
    "uber",
    "wolfram",
]
user_id = "performance_evaluation"


def read_excel(xlsx_path: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, keep_default_na=False, na_values=[])  # .iloc[23:]


def write_excel(df: pd.DataFrame, path: Optional[str] = "new_df.xlsx") -> None:
    df.to_excel(path, index_label=True)


def evaluate_with_llm(gt_answer: str, ai_answer: str) -> int:
    prompt = """\
**Perform the following steps:**
- Generate the intent for each sentence.
- Generate keywords corresponding to the intent in the sentences.
- Evaluate the similarity between sentences.

Sentence 1: {gt_answer}
Sentence 2: {ai_answer}

**Response:**
- Apply the steps to both sentence 1 and sentence 2.
- Generate in JSON format: {{"intent_1": "", "intent_2": "", "keywords_1": "", "keywords_2": "", "similarity": "similar/dissimilar"}}
- Just return JSON."""

    prompt = prompt.format(gt_answer=gt_answer, ai_answer=ai_answer)

    response = llm.query(
        prompt=prompt, model="gpt-4o", response_format=JSON_RESPONSE_FORMAT
    )

    similarity = json.loads(response).get("similarity", None)

    return 1 if similarity == "similar" else 0


def evaluate_slide_score(gt_answer: Union[str, float], ai_answer: str) -> int:
    if pd.isna(gt_answer):
        gt_answer = "None"
    try:
        num_a = float(gt_answer)
        num_b = float(ai_answer)
        return 1 if num_a == num_b else 0
    except ValueError:
        str_a = str(gt_answer).lower()
        str_b = str(ai_answer).lower()
        return 1 if str_a == str_b else 0


def process_row(row: pd.Series) -> Dict[str, Union[int, str]]:
    product_description = Presentation(list(json.loads(convert_to_json(row.product_description)).values()))
    user_messages, ai_messages = extract_messages(row.conversation.strip())
    chat_history = assemble_chat_history(user_messages, ai_messages)
    new_product_description = vsa.get_updated_product_description(product_description, chat_history)
    list_product_description = list(new_product_description.to_json().keys())

    current_answer = vsa(
        question=row.question,
        product_description=product_description,
        chat_history=chat_history,
        document_ids=document_id_list,
        user_id=user_id,
    )

    current_answer_text = current_answer.response
    current_answer_slide = current_answer.metadata.slide_id
    gt_slide_id_with_page = "page_" + str(int(row.slide_id)) if row.slide_id not in ["None", "none", ""] else "None"
    gt_answer = row.answer if gt_slide_id_with_page in list_product_description else "I don't know.$eos$"
    gt_slide_id = gt_slide_id_with_page if gt_slide_id_with_page in list_product_description else "None"

    generator_score = evaluate_with_llm(gt_answer=gt_answer, ai_answer=current_answer_text)
    slide_score = evaluate_slide_score(gt_answer=gt_slide_id, ai_answer=current_answer_slide)

    return {
        "ai_answer": current_answer_text,
        "ai_slide": current_answer_slide,
        "generator_score": generator_score,
        "slide_score": slide_score,
    }


def apply_function_concurrently(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["ai_slide"] = None
    df["gen_score"] = None
    df["slide_score"] = None

    with ThreadPoolExecutor(max_workers=int(8)) as executor:
        future_to_row = {
            executor.submit(process_row, row): i for i, row in df.iterrows()
        }

        for future in tqdm.tqdm(as_completed(future_to_row), total=len(df)):
            row_index = future_to_row[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Row {row_index} generated an exception: {exc}")
                df.at[row_index, "ai_answer"] = None
                df.at[row_index, "ai_slide"] = None
                df.at[row_index, "gen_score"] = None
                df.at[row_index, "slide_score"] = None
            else:
                df.at[row_index, "ai_answer"] = result["ai_answer"]
                df.at[row_index, "ai_slide"] = result["ai_slide"]
                df.at[row_index, "gen_score"] = result["generator_score"]
                df.at[row_index, "slide_score"] = result["slide_score"]

    return df


def apply_function_currently(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["ai_slide"] = None
    df["gen_score"] = None
    df["slide_score"] = None

    for row in tqdm.tqdm(df.iterrows(), total=len(df)):
        row_index = row[0]
        result = process_row(row[1])
        df.at[row_index, "ai_answer"] = result["ai_answer"]
        df.at[row_index, "ai_slide"] = result["ai_slide"]
        df.at[row_index, "gen_score"] = result["generator_score"]
        df.at[row_index, "slide_score"] = result["slide_score"]

        time.sleep(1)

    return df


def calculate_metrics(df: pd.DataFrame) -> None:
    gens_vs = df["gen_score"].dropna().to_list()
    gens_cs = df["slide_score"].dropna().to_list()

    final_gen_score_vs = sum(gens_vs) / len(gens_vs)
    final_gen_score_cs = sum(gens_cs) / len(gens_cs)

    print(
        f"The final achieved score for texts is: {round(final_gen_score_vs, 3)}"
    )
    print(
        f"The final achieved score for slide ids is: {round(final_gen_score_cs, 3)}"
    )

    true_positive = len(df[(df["slide_id"] == "None") & (df["slide_score"] == 1)])
    true_negative = len(df[(df["slide_id"] != "None") & (df["slide_score"] == 1)])
    false_positive = len(df[(df["slide_id"] == "None") & (df["slide_score"] == 0)])
    false_negative = len(df[(df["slide_id"] != "None") & (df["slide_score"] == 0)])

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    print(
        f"The final achieved precision for slides is: {round(precision, 3)}"
    )
    print(
        f"The final achieved recall for slides is: {round(recall, 3)}"
    )


def convert_to_json(input_string):
    references = re.findall(r'\{\{(.*?)\}\}', input_string, re.DOTALL)

    pages = {}

    for i, ref in enumerate(references, 1):
        ref_content = re.search(r'"reference_content": "(.*?)"', ref, re.DOTALL)
        if ref_content:
            pages[f"page_{i}"] = ref_content.group(1).replace('\n', ' ').strip()

    json_output = json.dumps(pages, indent=2)

    return json_output


def extract_messages(conversation: str) -> Tuple[List[str], List[str]]:
    user_pattern = r"(?:^|\n)(?:- ?)?user:? ?(.+?)(?=\n(?:- ?)?(?:user|ai agent)|$)"
    ai_pattern = r"(?:^|\n)(?:- ?)?ai agent:? ?(\{\{.+?\}\})"

    user_messages = re.findall(user_pattern, conversation, re.IGNORECASE | re.DOTALL)
    ai_messages = re.findall(ai_pattern, conversation, re.IGNORECASE | re.DOTALL)

    user_messages = [msg.strip() for msg in user_messages]
    ai_messages = [msg.replace("{{", "{").replace("}}", "}") for msg in ai_messages]

    return user_messages, ai_messages


def assemble_chat_history(user_messages: List[str], ai_messages: List[str]) -> ChatHistory:
    chat_history = []
    if len(ai_messages) == len(user_messages) + 1:
        user_messages.insert(0, "")
    elif len(ai_messages) >= len(user_messages):
        user_messages = user_messages[:len(ai_messages)]
    else:
        ai_messages = ai_messages[:len(user_messages)]

    if len(ai_messages) == len(user_messages):
        for ai, user in zip(ai_messages, user_messages):
            json_ai = json.loads(ai)
            slide_id = "page_" + str(int(json_ai["1_used_reference_id"])) if json_ai["1_used_reference_id"] not in ["None", "none", ""] else "None"
            answer = json_ai["2_text"]
            chat_history.append(Dialogue(user=user, assistant=answer, slide_id=slide_id))
    else:
        raise ValueError("Error in messages equality")

    return ChatHistory(chat_history)


def main():
    eval_data_path = os.path.abspath(
        pathlib.Path(__file__).parent.parent / "files" / "perf_eval_visual_data.xlsx"
    )
    eval_df = read_excel(eval_data_path)

    new_df = apply_function_concurrently(eval_df)

    write_excel(new_df)
    calculate_metrics(new_df)


if __name__ == "__main__":
    main()
