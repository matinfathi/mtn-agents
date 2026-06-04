from mtn_agents.constants import OPENAI_EMBEDDING_MODEL_2, LATEST_EMBEDDING_MODEL
from mtn_agents.update_users import update_embedding_for_user
from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.chains import MasterChain

from datetime import datetime
import unittest
import pathlib
import time
import os


class TestEmbeddingModel(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime('%B_%d_%Y_%H:%M:%S')

    namespace = f"test_embedding_{formatted_time}"
    document_id = "deep learning"
    pc_instance = PineconeIndex()

    local_path_document = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "test_doc.pdf"
    )

    master_chain = MasterChain()

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_file_to_index_pinecone(
            local_path=cls.local_path_document,
            user_id=cls.namespace,
            document_id=cls.document_id,
            embedding_model=OPENAI_EMBEDDING_MODEL_2
        )
        while True:
            info_dict = cls.pc_instance.freshness()
            list_namespaces = list(info_dict["namespaces"].keys())
            if cls.namespace in list_namespaces:
                break

            time.sleep(2)

    def test_00_ask_question(self):
        answer = self.master_chain(
            question="What is deep learning?",
            user_id=self.namespace,
            document_ids=[self.document_id],
            embedding_model=OPENAI_EMBEDDING_MODEL_2,
        )

        response = answer["response"]
        references = answer["metadata"]["references"]

        self.assertIsInstance(response, str)
        self.assertTrue(response.strip(), "Response should not be empty")
        self.assertIsInstance(references, list)
        self.assertTrue(references, "References should not be empty")
        self.assertIsInstance(references[0], dict)
        self.assertIsInstance(references[0]["score"], float)
        self.assertGreater(references[0]["score"], 0.7)

    def test_01_ask_question_2(self):
        update_embedding_for_user(
            user_id=self.namespace,
            embedding_model=LATEST_EMBEDDING_MODEL,
        )
        time.sleep(5)
        answer = self.master_chain(
            question="What is deep learning?",
            user_id=self.namespace,
            document_ids=[self.document_id],
            embedding_model=LATEST_EMBEDDING_MODEL,
        )

        response = answer["response"]
        references = answer["metadata"]["references"]

        self.assertIsInstance(response, str)
        self.assertTrue(response.strip(), "Response should not be empty")
        self.assertIsInstance(references, list)
        self.assertTrue(references, "References should not be empty")
        self.assertIsInstance(references[0], dict)
        self.assertIsInstance(references[0]["score"], float)
        self.assertGreater(references[0]["score"], 0.7)

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.namespace)


if __name__ == "__main__":
    unittest.main()
