from mtn_agents.constants import (
    GREET_PICKLE_PATH,
    MODEL_PICKLE_PATH,
    MTN_AGENTS_PICKLE_PATH,
    GREET_AR_PICKLE_PATH,
    MODEL_AR_PICKLE_PATH,
    MTN_AGENTS_AR_PICKLE_PATH,
)
from mtn_agents.utils import hex_uuid
from mtn_agents.llm import OpenAILLM

from typing import Dict, Union, List
import pandas as pd
import pickle

llm = OpenAILLM()


def read_xlsx(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def convert_xlsx_to_pc_format(current_row: pd.Series, embed: List[float]) -> Dict[str, Union[Dict[str, str], str]]:
    return {
        "id": hex_uuid(),
        "values": embed,
        "metadata": {
            "text": f"""{{"question": "{current_row.question}", "answer": "{current_row.answer}"}}""",
            "document_id": current_row.route,
            "cache_id": current_row.route,
        }
    }


sr_data = read_xlsx("")

questions = sr_data["question"].tolist()
question_embeddings = llm.embedding(list_texts=questions)

greet, model, company = [], [], []
greet_ar, model_ar, company_ar = [], [], []

for idx, (i, row) in enumerate(sr_data.iterrows()):
    if row.route == "greet":
        greet.append(convert_xlsx_to_pc_format(row, question_embeddings[idx]))
    elif row.route == "model":
        greet.append(convert_xlsx_to_pc_format(row, question_embeddings[idx]))
    elif row.route == "company":
        greet.append(convert_xlsx_to_pc_format(row, question_embeddings[idx]))
    elif row.route == "greet_ar":
        greet.append(convert_xlsx_to_pc_format(row, question_embeddings[idx]))
    elif row.route == "model_ar":
        greet.append(convert_xlsx_to_pc_format(row, question_embeddings[idx]))
    elif row.route == "company_ar":
        greet.append(convert_xlsx_to_pc_format(row, question_embeddings[idx]))


with open(GREET_PICKLE_PATH, 'wb') as file:
    pickle.dump(greet, file)

with open(MODEL_PICKLE_PATH, 'wb') as file:
    pickle.dump(model, file)

with open(MTN_AGENTS_PICKLE_PATH, 'wb') as file:
    pickle.dump(company, file)

with open(GREET_AR_PICKLE_PATH, 'wb') as file:
    pickle.dump(greet_ar, file)

with open(MODEL_AR_PICKLE_PATH, 'wb') as file:
    pickle.dump(model_ar, file)

with open(MTN_AGENTS_AR_PICKLE_PATH, 'wb') as file:
    pickle.dump(company_ar, file)
