from mtn_agents.llm import OpenAILLM
from mtn_agents.chains import SemanticRouterChain
from mtn_agents.utils import Language

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, Any, Union
import pandas as pd
import numpy as np
import asyncio
import pathlib
import tqdm
import time
import json
import os

rag = SemanticRouterChain()
llm = OpenAILLM()

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
]
user_id = "performance_evaluation"


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
        prompt=prompt, model="gpt-4-turbo", response_format={"type": "json_object"}
    )

    similarity = json.loads(response).get("similarity", None)

    return 1 if similarity == "similar" else 0


def evaluate_retriever(current_answer: Dict[str, Any], row: pd.Series) -> int:
    ref_id_list = [item.get("id", "") for item in current_answer["reference_metadata"]]

    if row.target_chunk_id in ref_id_list:
        return 1
    else:
        return 0


def evaluate_generator(current_answer: Dict[str, str], row: pd.Series) -> int:
    return evaluate_with_llm(
        gt_answer=row.answer,
        ai_answer=current_answer["response"],
    )


def evaluate_semantic_router(current_answer: Dict[str, str], row: pd.Series) -> int:
    if current_answer["intent"] == row.route:
        return 1
    else:
        return 0


def process_row(row: pd.Series) -> Dict[str, Union[int, str]]:
    language = Language.ARABIC if row.lang == "ar" else Language.ENGLISH
    start_time = time.perf_counter()
    current_answer = rag.qa(
        chat_history="",
        question=row.question,
        document_ids=document_id_list,
        user_id=user_id,
        language=language,
    )
    end_time = time.perf_counter()
    latency = round(end_time - start_time, 3)

    if row.route != "reference":
        semantic_router_score = evaluate_semantic_router(current_answer, row)
        return {
            "ai_route": current_answer["intent"],
            "ai_answer": current_answer["response"],
            "ai_ref_list": [
                item.get("id", "") for item in current_answer["reference_metadata"]
            ],
            "retriever_score": np.nan,
            "generator_score": np.nan,
            "semantic_router_score": semantic_router_score,
            "latency": latency,
        }
    else:
        semantic_router_score = evaluate_semantic_router(current_answer, row)
        retriever_score = (
            evaluate_retriever(current_answer, row)
            if not pd.isna(row.target_chunk_id)
            else np.nan
        )
        generator_score = evaluate_generator(current_answer, row)

        return {
            "ai_route": current_answer["intent"],
            "ai_answer": current_answer["response"],
            "ai_ref_list": [
                item.get("id", "") for item in current_answer["reference_metadata"]
            ],
            "retriever_score": retriever_score,
            "generator_score": generator_score,
            "semantic_router_score": semantic_router_score,
            "latency": latency,
        }


def apply_function_concurrently(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["list_refs"] = None
    df["ai_route"] = None
    df["ret_score"] = None
    df["gen_score"] = None
    df["sem_score"] = None
    df["latency"] = None

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
                df.at[row_index, "list_refs"] = None
                df.at[row_index, "ai_route"] = None
                df.at[row_index, "ret_score"] = None
                df.at[row_index, "gen_score"] = None
                df.at[row_index, "sem_score"] = None
                df.at[row_index, "latency"] = None
            else:
                df.at[row_index, "ai_answer"] = result["ai_answer"]
                df.at[row_index, "list_refs"] = result["ai_ref_list"]
                df.at[row_index, "ai_route"] = result["ai_route"]
                df.at[row_index, "ret_score"] = result["retriever_score"]
                df.at[row_index, "gen_score"] = result["generator_score"]
                df.at[row_index, "sem_score"] = result["semantic_router_score"]
                df.at[row_index, "latency"] = result["latency"]

    return df


async def apply_function_concurrently_async(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["list_refs"] = None
    df["ai_route"] = None
    df["ret_score"] = None
    df["gen_score"] = None
    df["sem_score"] = None
    df["latency"] = None

    semaphore = asyncio.Semaphore(8)

    # tasks = [asyncio.create_task()]

    return df


def apply_function_currently(df: pd.DataFrame) -> pd.DataFrame:
    df["ai_answer"] = None
    df["list_refs"] = None
    df["ai_route"] = None
    df["ret_score"] = None
    df["gen_score"] = None
    df["sem_score"] = None
    df["latency"] = None

    for row in tqdm.tqdm(df.itertuples(), total=len(df)):
        row_index = row[0]
        result = process_row(row)
        df.at[row_index, "ai_answer"] = result["ai_answer"]
        df.at[row_index, "list_refs"] = result["ai_ref_list"]
        df.at[row_index, "ai_route"] = result["ai_route"]
        df.at[row_index, "ret_score"] = result["retriever_score"]
        df.at[row_index, "gen_score"] = result["generator_score"]
        df.at[row_index, "sem_score"] = result["semantic_router_score"]
        df.at[row_index, "latency"] = result["latency"]

    return df


def calculate_metrics(df: pd.DataFrame) -> None:
    df_en = df[df["lang"] == "en"]
    df_ar = df[df["lang"] == "ar"]

    df_en_ref = df_en[df_en["route"] == "reference"]["latency"].dropna().to_list()
    df_en_no_ref = df_en[df_en["route"] != "reference"]["latency"].dropna().to_list()

    df_ar_ref = df_ar[df_ar["route"] == "reference"]["latency"].dropna().to_list()
    df_ar_no_ref = df_ar[df_ar["route"] != "reference"]["latency"].dropna().to_list()

    sems_en = df_en["sem_score"].dropna().to_list()
    rets_en = df_en["ret_score"].dropna().to_list()
    gens_en = df_en["gen_score"].dropna().to_list()

    sems_ar = df_ar["sem_score"].dropna().to_list()
    rets_ar = df_ar["ret_score"].dropna().to_list()
    gens_ar = df_ar["gen_score"].dropna().to_list()

    final_sem_score_en = sum(sems_en) / len(sems_en)
    final_ret_score_en = sum(rets_en) / len(rets_en)
    final_gen_score_en = sum(gens_en) / len(gens_en)

    final_sem_score_ar = sum(sems_ar) / len(sems_ar)
    final_ret_score_ar = sum(rets_ar) / len(rets_ar)
    final_gen_score_ar = sum(gens_ar) / len(gens_ar)

    final_latency_en_ref = sum(df_en_ref) / len(df_en_ref)
    final_latency_en_no_ref = sum(df_en_no_ref) / len(df_en_no_ref)
    final_latency_ar_ref = sum(df_ar_ref) / len(df_ar_ref)
    final_latency_ar_no_ref = sum(df_ar_no_ref) / len(df_ar_no_ref)

    print("English:\n")
    print(
        f"The final achieved score for English semantic router is: {round(final_sem_score_en, 3)}"
    )
    print(
        f"The final achieved score for English retriever is: {round(final_ret_score_en, 3)}"
    )
    print(
        f"The final achieved score for English generator is: {round(final_gen_score_en, 3)}"
    )
    print(
        f"The average latency for English RagV2 with cache is: {round(final_latency_en_no_ref, 3)}"
    )
    print(
        f"The average latency for English RagV2 with out cache is: {round(final_latency_en_ref, 3)}"
    )
    print("\n--------------\nArabic:\n")
    print(
        f"The final achieved score for Arabic semantic router is: {round(final_sem_score_ar, 3)}"
    )
    print(
        f"The final achieved score for Arabic retriever is: {round(final_ret_score_ar, 3)}"
    )
    print(
        f"The final achieved score for Arabic generator is: {round(final_gen_score_ar, 3)}"
    )
    print(
        f"The average latency for Arabic RagV2 with cache is: {round(final_latency_ar_no_ref, 3)}"
    )
    print(
        f"The average latency for Arabic RagV2 with out cache is: {round(final_latency_ar_ref, 3)}"
    )


def main():
    eval_data_path = os.path.abspath(
        pathlib.Path(__file__).parent.parent / "files" / "ev_dataset.xlsx"
    )
    eval_df = read_excel(eval_data_path)

    new_df = apply_function_currently(eval_df)

    calculate_metrics(new_df)
    write_excel(new_df)


if __name__ == "__main__":
    main()
