from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.features import auto_generate_slide_concurrent

from datetime import datetime
import unittest
import pathlib
import time
import os


class TestFeatures(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime('%B_%d_%Y_%H:%M:%S')

    agent_uuid = "11-22-33-44"

    local_path_pptx = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "test_pptx_confluence.pptx"
    )
    local_path_additional = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "confluence_additional_data.txt"
    )
    namespace = f"test_auto_slide_{formatted_time}"
    document_id = "confluence"
    pc_instance = PineconeIndex()

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_file_to_index_pinecone(
            local_path=cls.local_path_additional,
            user_id=cls.namespace,
            document_id=cls.document_id,
        )
        while True:
            info_dict = cls.pc_instance.freshness()
            list_namespaces = list(info_dict["namespaces"].keys())
            if cls.namespace in list_namespaces:
                break

            time.sleep(2)

    def test_00_auto_slide(self):
        data = auto_generate_slide_concurrent(
            slide_path=self.local_path_pptx,
            user_id=self.namespace,
            document_ids=[self.document_id],
            agent_uuid=self.agent_uuid,
        )

        self.assertIsInstance(data, dict)
        self.assertEqual(len(data), 9)
        self.assertEqual(list(data.keys())[0], "page_1")

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.namespace)


if __name__ == "__main__":
    unittest.main()
