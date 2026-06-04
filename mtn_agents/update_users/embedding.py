import time

from mtn_agents.llm import OpenAILLM
from mtn_agents.retriever import PineconeIndex
from mtn_agents.utils import hex_uuid
from mtn_agents.constants import LATEST_EMBEDDING_MODEL
# from my_tests import my_test_base


def update_embedding_for_user(user_id: str, embedding_model: str = LATEST_EMBEDDING_MODEL):
    llm_instance = OpenAILLM()
    pc_instance = PineconeIndex()
    pc_instance.load_index()

    paginated_token = None
    list_new_records = []

    while True:
        result_list = pc_instance.index.list_paginated(
            limit=100,
            namespace=user_id,
            pagination_token=paginated_token,
        )

        list_ids = [v.id for v in result_list.vectors]

        if list_ids:
            list_records_raw = pc_instance.index.fetch(list_ids, namespace=user_id)
            list_records = list(list_records_raw["vectors"].values())

            list_texts = [record["metadata"]["text"] for record in list_records]

            embeddings = llm_instance.embedding(list_texts=list_texts, model=embedding_model)

            for record, embedding in zip(list_records, embeddings):
                record["metadata"]["embedding_model"] = embedding_model
                list_new_records.append(
                    {
                        "id": hex_uuid()[:12],
                        "values": embedding,
                        "metadata": record["metadata"],
                    }
                )

            if result_list.pagination is None:
                break
            else:
                paginated_token = result_list.pagination.next

        else:
            break

        time.sleep(1)

    pc_instance.detailed_upsert(vectors=list_new_records, user_id=user_id)


# update_embedding_for_user(user_id="100")  # test_gulf_bank
