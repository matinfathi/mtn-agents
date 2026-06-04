from mtn_agents.upload_reference import upload_file_to_index_pinecone, upload_url_to_index_pinecone
from mtn_agents.features import auto_generate_landing_presentation
from mtn_agents.retriever import PineconeIndex
from mtn_agents.agents import SiteSageAgent
from mtn_agents.utils import ChatHistory, Dialogue

from types import GeneratorType
from datetime import datetime
import unittest
import pathlib
import time
import os


class TestSiteSageAgent(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime("%B_%d_%Y_%H:%M:%S")

    url = "https://en.wikipedia.org/wiki/Deep_learning"
    url_2 = "https://en.wikipedia.org/wiki/Zeng_Xisheng"

    user_id = f"test_site_sage_agent_{formatted_time}"
    agent_uuid = "22-33-44-55"
    document_ids = ["mtn_agents_site", "mtn_agents_reference", "wikipedia"]

    agent = SiteSageAgent()
    pc_instance = PineconeIndex()

    product_description = None

    local_path = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "mtn_agents_Reference.pdf"
    )

    @classmethod
    def setUpClass(cls):
        _ = upload_url_to_index_pinecone(
            url_address=cls.url,
            user_id=cls.user_id,
            document_id=cls.document_ids[0],
        )

        cls.product_description = auto_generate_landing_presentation(
            url=cls.url,
            agent_uuid=cls.agent_uuid,
            user_id=cls.user_id,
            document_ids=cls.document_ids,
        )
        _ = upload_file_to_index_pinecone(
            local_path=cls.local_path,
            user_id=cls.user_id,
            document_id=cls.document_ids[1],
            cache_q=False,
        )

        _ = upload_url_to_index_pinecone(
            url_address=cls.url_2,
            user_id=cls.user_id,
            document_id=cls.document_ids[2],
        )

        while True:
            time.sleep(2)
            info_dict = cls.pc_instance.freshness()
            list_namespaces = list(info_dict["namespaces"].keys())
            if cls.user_id in list_namespaces:
                break

    def test_00_start_conversation(self):
        agent_response = self.agent(
            product_description=self.product_description,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
        )

        self.assertIsInstance(agent_response, dict)
        self.assertIsInstance(agent_response["response"], GeneratorType)

        full_response = "".join(
            [token if type(token) is str else "" for token in agent_response["response"]]
        )

        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertIn(agent_response["metadata"]["slide_id"].lower(), ["none", "page_1"])
        self.assertIn(agent_response["metadata"]["chain"], ["master_chain", "inprompt_chain"])
        self.assertIsInstance(agent_response["metadata"]["prompt"], str)
        self.assertGreater(len(agent_response["metadata"]["prompt"]), 10)
        self.assertIsInstance(agent_response["metadata"]["url"], str)
        # self.assertIn("mtn_agents", agent_response["metadata"]["url"])
        self.assertEqual(agent_response["metadata"]["document_id"], "current")

    def test_01_inprompt_chain_question(self):
        chat_history = ChatHistory(
            [Dialogue(user="", assistant="Hello!$eos$ This is Joe.$eos$ How are you today?$eos$")]
        )

        question = "tell me about mtn_agents pricing?"

        agent_response = self.agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Joe",
            agent_uuid=self.agent_uuid,
            main_document_id=self.document_ids[0],
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join(
            [token if type(token) is str else "" for token in agent_response["response"]]
        )

        self.assertIsInstance(full_response, str)
        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertEqual(agent_response["metadata"]["chain"], "master_chain")
        self.assertIsInstance(agent_response["metadata"]["prompt"], str)
        self.assertGreater(len(agent_response["metadata"]["prompt"]), 10)
        self.assertIsInstance(agent_response["metadata"]["url"], str)

    def test_02_master_chain_question(self):
        chat_history = ChatHistory(
            [Dialogue(user="", assistant="Hello!$eos$ This is Joe.$eos$ How are you today?$eos$")]
        )

        question = "ُWho are the creators of mtn_agents?"

        agent_response = self.agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Joe"
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join(
            [token if type(token) is str else "" for token in agent_response["response"]]
        )

        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertIn(agent_response['metadata']['chain'], ["master_chain"])
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)
        self.assertIsInstance(agent_response['metadata']['url'], str)
        self.assertEqual(agent_response['metadata']['url'], "None")
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertIsInstance(agent_response['metadata']['references'][0]['text'], str)
        self.assertEqual(agent_response["metadata"]["document_id"], "None")

    def test_03_master_chain_question(self):
        chat_history = ChatHistory(
            [Dialogue(user="", assistant="Hello!$eos$ This is Joe.$eos$ How are you today?$eos$")]
        )

        question = "Who is Zeng Xisheng?"

        agent_response = self.agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Joe"
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join(
            [token if type(token) is str else "" for token in agent_response["response"]]
        )

        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertIn(agent_response['metadata']['chain'], ["master_chain"])
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)
        self.assertIsInstance(agent_response['metadata']['url'], str)
        self.assertEqual(agent_response['metadata']['url'], "https://en.wikipedia.org/wiki/Zeng_Xisheng")
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertEqual(agent_response['metadata']['document_id'], self.document_ids[2])
        self.assertIsInstance(agent_response['metadata']['references'][0]['text'], str)

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.user_id)


# if __name__ == "__main__":
#     unittest.main()
