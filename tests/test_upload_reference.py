from mtn_agents.upload_reference import (
    upload_file_to_index_pinecone,
    upload_url_to_index_pinecone,
)
from mtn_agents.retriever import PineconeIndex

from datetime import datetime
import unittest
import time
import pathlib
import os


class TestUploadReference(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime('%B_%d_%Y_%H:%M:%S')

    document_id = "unittest_doc"
    namespace1 = f"test_upload_reference_file_{formatted_time}"
    namespace2 = f"test_upload_reference_url_{formatted_time}"
    local_path = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "ML_DL.pdf"
    )
    pc_instance = PineconeIndex()
    question = "What is Deep learning??"
    url_address = "https://en.wikipedia.org/wiki/Deep_learning"
    question_url = "What is mtn_agents?"

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_file_to_index_pinecone(
            local_path=cls.local_path,
            user_id=cls.namespace1,
            document_id=cls.document_id,
            cache_q=False,
            is_slide=True,
        )
        cls.pinecone_data_url = upload_url_to_index_pinecone(
            url_address=cls.url_address,
            user_id=cls.namespace2,
            document_id=cls.document_id,
            cache_q=False,
            async_mode=False,
        )
        while True:
            info_dict = cls.pc_instance.freshness()
            list_namespaces = list(info_dict["namespaces"].keys())
            if cls.namespace1 in list_namespaces and cls.namespace2 in list_namespaces:
                break

            time.sleep(2)

    def test_00_upload_pdf_to_index(self):
        self.assertGreater(len(self.pinecone_data_file), 0)

    def test_01_namespace_creation(self):
        info_dict = self.pc_instance.freshness()
        list_namespaces = list(info_dict["namespaces"].keys())
        self.assertIn(self.namespace1, list_namespaces)

    def test_02_query_reference(self):
        filter_dict_ref = {
            "document_id": {"$in": [self.document_id]},
            "cache_id": {"$exists": False},
        }
        query_data = self.pc_instance.detailed_query(
            input_texts=[self.question],
            user_id=self.namespace1,
            filter_dict=filter_dict_ref,
            top_k=10,
        )
        texts = [item["metadata"]["text"] for item in query_data["matches"]]
        self.assertIn(len(query_data["matches"]), list(range(7)))
        self.assertIn(
            self.pinecone_data_file[0]["text"],
            texts,
        )
        self.assertEqual(query_data["matches"][0]["metadata"]["file_extension"], "pdf")
        self.assertEqual(query_data["matches"][0]["metadata"]["chunk_type"], "slide")

    def test_03_upload_url(self):
        filter_dict_ref = {
            "document_id": {"$in": [self.document_id]},
            "cache_id": {"$nin": [self.document_id]},
        }
        query_data = self.pc_instance.detailed_query(
            input_texts=[self.question],
            user_id=self.namespace2,
            filter_dict=filter_dict_ref,
            top_k=10,
        )
        self.assertGreater(len(query_data["matches"]), 0)
        self.assertEqual(query_data["matches"][0]["metadata"]["file_extension"], "url")

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.namespace1)
        cls.pc_instance.delete_user_id(user_id=cls.namespace2)


if __name__ == "__main__":
    unittest.main()
