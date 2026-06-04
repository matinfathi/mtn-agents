from mtn_agents.upload_reference import upload_url_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.features import auto_generate_landing_presentation

from datetime import datetime
import unittest
import time


class TestFeaturesAutoURL(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime('%B_%d_%Y_%H:%M:%S')
    url_address = "https://en.wikipedia.org/wiki/Deep_learning"
    namespace = f"test_auto_url_{formatted_time}"
    agent_uuid = "11-22-33-44"
    document_id = "mtn_agents"
    pc_instance = PineconeIndex()

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_url_to_index_pinecone(
            url_address=cls.url_address,
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
        data = auto_generate_landing_presentation(
            url=self.url_address,
            agent_uuid=self.agent_uuid,
            user_id=self.namespace,
        )

        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 5)
        self.assertGreater(len(data["page_1"]["url"]), 1)
        self.assertIsInstance(data["page_1"]["url"], str)
        self.assertEqual(list(data.keys())[0], "page_1")

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.namespace)


# if __name__ == "__main__":
#     unittest.main()
