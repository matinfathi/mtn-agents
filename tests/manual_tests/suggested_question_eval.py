from mtn_agents.llm import OpenAILLM
from mtn_agents.features import generate_suggested_questions_with_answer
from mtn_agents.utils import Language

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List
import pandas as pd
import pathlib
import tqdm
import json
import os


def read_excel(xlsx_path: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path)


def write_excel(df: pd.DataFrame, path: Optional[str] = "new_df.xlsx") -> None:
    df.to_excel(path, index=False)


def eval_generated_suggested_question(suggested_q: Dict[str, str], text: str) -> int:
    prompt = """\
text:
```{text}```

question: {question}
answer: {answer}

Is the answer of the question in the text? answer in JSON with yes/no: {{"answer": "yes/no"}}"""

    prompt = prompt.format(
        text=text, question=suggested_q["question"], answer=suggested_q["answer"]
    )

    answer = OpenAILLM().query(
        prompt=prompt, model="gpt-4-turbo", response_format={"type": "json_object"}
    )

    return 1 if json.loads(answer)["answer"].lower() == "yes" else 0


def process_row(row: pd.Series) -> List[Dict[str, str]]:
    if pd.isna(row.text):
        return []

    lang = (
        Language.ARABIC.description
        if Language.ARABIC.code == row.lang
        else Language.ENGLISH.description
    )
    current_answer = generate_suggested_questions_with_answer(
        summary_text=row.text, num_question=2, lang=lang
    )

    return_list = []

    for item in current_answer:
        evaluation = eval_generated_suggested_question(item, row.text)
        return_list.append(
            {
                "lang": row.lang,
                "text": row.text,
                "question": item["question"],
                "answer": item["answer"],
                "score": evaluation,
            }
        )

    return return_list


def apply_function_concurrently(df: pd.DataFrame) -> pd.DataFrame:
    new_df = pd.DataFrame([], columns=["lang", "text", "question", "answer", "score"])

    with ThreadPoolExecutor(max_workers=int(os.cpu_count()) + 4) as executor:
        future_to_row = {
            executor.submit(process_row, row): i for i, row in df.iterrows()
        }

        for future in tqdm.tqdm(as_completed(future_to_row), total=len(df)):
            row_index = future_to_row[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"Row {row_index} generated an exception: {exc}")
                new_df.loc[row_index] = [None, None, None, None, None]
            else:
                for item in result:
                    new_df.loc[len(new_df)] = [
                        item["lang"],
                        item["text"],
                        item["question"],
                        item["answer"],
                        item["score"],
                    ]

    return new_df


def apply_function_currently(df: pd.DataFrame) -> pd.DataFrame:
    new_df = pd.DataFrame([], columns=["lang", "text", "question", "answer", "score"])

    for row in tqdm.tqdm(df.itertuples(), total=len(df)):
        result = process_row(row)

        for item in result:
            new_df.loc[len(new_df)] = [
                item["lang"],
                item["text"],
                item["question"],
                item["answer"],
                item["score"],
            ]

    return new_df


def calculate_metrics(df: pd.DataFrame):
    df_en = df[df["lang"] == "en"]
    df_ar = df[df["lang"] == "ar"]

    en_scores = df_en["score"].dropna().to_list()
    ar_scores = df_ar["score"].dropna().to_list()

    final_en_scores = sum(en_scores) / len(en_scores)
    final_ar_scores = sum(ar_scores) / len(ar_scores)

    print(
        f"The achieved score for English suggested questions is: {round(final_en_scores, 3)}"
    )
    print(
        f"The achieved score for Arabic suggested questions is: {round(final_ar_scores, 3)}"
    )


def main():
    eval_data_path = os.path.abspath(
        pathlib.Path(__file__).parent.parent / "files" / "ev_dataset.xlsx"
    )
    eval_df = read_excel(eval_data_path)

    new_df = apply_function_concurrently(eval_df)

    calculate_metrics(new_df)
    write_excel(new_df, path="suggested_q_eval.xlsx")


if __name__ == "__main__":
    main()
