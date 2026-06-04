from mtn_agents.prompt_templates import prompt_voice_sales_agent, prompt_customer_support_agent
from mtn_agents.chains import InPromptChain
from mtn_agents.llm import OpenAILLM

import pandas as pd
import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Union, Dict
import pathlib
import json
import os


llm = OpenAILLM()
inprompt_chain = InPromptChain()


def read_excel(xlsx_path: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)


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
        prompt=prompt, model="gpt-4o", response_format={"type": "json_object"}
    )

    similarity = json.loads(response).get("similarity", None)

    return 1 if similarity == "similar" else 0


def process_row(row: pd.Series) -> Dict[str, Union[int, str]]:
    required_texts = {
        "product_description": row.product_description,
        "additional_information": "",
    }
    current_answer = inprompt_chain(
        question=row.question,
        chat_history_str=row.conversation,
        required_texts=required_texts,
        prompt=prompt_voice_sales_agent if row.agent_type == "Voice Sales Agent" else prompt_customer_support_agent,
    )["response"]

    generator_score = evaluate_with_llm(gt_answer=row.answer, ai_answer=current_answer)

    return {
        "ai_answer": current_answer,
        "generator_score": generator_score,
    }


def apply_function_concurrently(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["gen_score"] = None

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
                df.at[row_index, "gen_score"] = None
            else:
                df.at[row_index, "ai_answer"] = result["ai_answer"]
                df.at[row_index, "gen_score"] = result["generator_score"]

    return df


def apply_function_currently(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["gen_score"] = None

    for row in tqdm.tqdm(df.iterrows(), total=len(df)):
        row_index = row[0]
        result = process_row(row[1])
        df.at[row_index, "ai_answer"] = result["ai_answer"]
        df.at[row_index, "gen_score"] = result["generator_score"]

    return df


def calculate_metrics(df: pd.DataFrame) -> None:
    df_vs = df[df["agent_type"] == "Voice Sales Agent"]
    df_cs = df[df["agent_type"] == "Voice Customer Support"]

    gens_vs = df_vs["gen_score"].dropna().to_list()
    gens_cs = df_cs["gen_score"].dropna().to_list()

    final_gen_score_vs = sum(gens_vs) / len(gens_vs)
    final_gen_score_cs = sum(gens_cs) / len(gens_cs)

    print(
        f"The final achieved score for Voice Sale Agent is: {round(final_gen_score_vs, 3)}"
    )
    print(
        f"The final achieved score for Voice Customer Support Agent is: {round(final_gen_score_cs, 3)}"
    )


def main():
    eval_data_path = os.path.abspath(
        pathlib.Path(__file__).parent.parent / "files" / "perf_eval_voice_data.xlsx"
    )
    eval_df = read_excel(eval_data_path)

    new_df = apply_function_concurrently(eval_df)

    write_excel(new_df)
    calculate_metrics(new_df)


if __name__ == "__main__":
    main()
