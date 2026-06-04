from mtn_agents.prompt_templates import prompt_visual_sales_agent_v35
from mtn_agents.constants import JSON_RESPONSE_FORMAT
from mtn_agents.utils import Presentation
from mtn_agents.chains import InPromptChain
from mtn_agents.llm import OpenAILLM
from my_tests import my_test_base

import pandas as pd
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, Dict
import pathlib
import json
import time
import re
import os

llm = OpenAILLM()
inprompt_chain = InPromptChain(model="gpt")


def read_excel(xlsx_path: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, keep_default_na=False, na_values=[])


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
    covered_slides = list(set(re.findall(r'"1_used_reference_id": "(\d+)"', row.conversation)))

    required_texts = {
        "product_description": product_description.to_json(to_str=True),
        "additional_information": "",
        "covered_slides": covered_slides,
        "agent_name": "Jamie",
        "action_item_rules": "",
    }
    current_answer = inprompt_chain(
        question=row.question,
        chat_history_str=row.conversation,
        required_texts=required_texts,
        prompt=prompt_visual_sales_agent_v35,
        response_format=JSON_RESPONSE_FORMAT
    )["response"]

    current_answer_json = json.loads(current_answer)
    current_answer_text = current_answer_json["2_text"]
    current_answer_slide = current_answer_json["1_used_reference_id"]

    generator_score = evaluate_with_llm(gt_answer=row.answer, ai_answer=current_answer_text)
    slide_score = evaluate_slide_score(gt_answer=row.slide_id, ai_answer=current_answer_slide)

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
