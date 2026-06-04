from mtn_agents.retriever import PineconeIndex
from mtn_agents.constants import LATEST_EMBEDDING_MODEL

from pinecone.core.openapi.data.model.scored_vector import ScoredVector

from datetime import datetime
from typing import List
import unittest
import time


class TestPineconeIndex(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime('%B_%d_%Y__%H:%M:%S')

    pinecone_instance = PineconeIndex()
    data = None
    document_id = "test_doc1"
    namespace = f"test_pinecone_{formatted_time}"

    @classmethod
    def setUpClass(cls):
        cls.data = [
            {
                "content": "Hello world!",
                "document_id": cls.document_id,
                "user_id": cls.namespace,
                "page_number": 1,
            }
        ]
        cls.metadata = {
            "document_id": cls.document_id,
            "user_id": cls.namespace,
            "page_number": 1,
            "text": cls.data[0]["content"],
        }

    def test_00_pinecone_init(self):
        self.assertIsNotNone(self.pinecone_instance)

    def test_01_pinecone_prepare_data(self):
        prepared_data = self.pinecone_instance.prepare_data_for_pinecone(self.data, embedding_model=LATEST_EMBEDDING_MODEL)
        prepared_data = prepared_data[0]
        self.assertIn("id", prepared_data.keys())
        self.assertIn("values", prepared_data.keys())
        self.assertIn("metadata", prepared_data.keys())
        self.assertEqual(prepared_data["metadata"], self.metadata)
        self.assertIsInstance(prepared_data["id"], str)
        self.assertIsInstance(prepared_data["values"], List)

    def test_02_pinecone_upsert(self):
        try:
            prepared_data = self.pinecone_instance.prepare_data_for_pinecone(self.data, embedding_model=LATEST_EMBEDDING_MODEL)
            self.pinecone_instance.detailed_upsert(
                vectors=prepared_data, user_id=self.namespace
            )
            time.sleep(5)
        except Exception as e:
            raise Exception(f"Upsert data into pinecone failed. Error {e}") from e

    def test_03_pinecone_query(self):
        text = self.data[0].get("content", self.data[0].get("text", ""))
        filter_dict = {"document_id": {"$in": [self.document_id]}}
        top_k = 1
        time.sleep(15)
        query = self.pinecone_instance.detailed_query(
            input_texts=[text],
            user_id=self.namespace,
            filter_dict=filter_dict,
            top_k=top_k,
            embedding_model=LATEST_EMBEDDING_MODEL,
        )
        query = query.get("matches", [])
        self.assertGreater(len(query), 0)
        self.assertIsInstance(query[0], ScoredVector)

        query = query[0].to_dict()
        self.assertIn("id", query.keys())
        self.assertIn("values", query.keys())
        self.assertIn("metadata", query.keys())
        self.assertEqual(text, query["metadata"]["text"])

    @classmethod
    def tearDownClass(cls):
        time.sleep(5)
        cls.pinecone_instance.delete_user_id(user_id=cls.namespace)


if __name__ == "__main__":
    unittest.main()
