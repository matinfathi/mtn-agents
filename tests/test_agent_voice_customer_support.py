from mtn_agents.agents import VoiceCustomerSupportAgent
from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.utils import (
    ChatHistory,
    Dialogue,
)

from types import GeneratorType
from datetime import datetime
import unittest
import pathlib
import time
import os


class TestCustomerSupportAgents(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime("%B_%d_%Y_%H:%M:%S")

    product_description = """\
Introducing SoleStride, a dynamic footwear company dedicated to crafting premium shoes that marry style with comfort. \
Founded by avid runners and footwear enthusiasts Emily Parker and Alex Chen, SoleStride was born out of a shared \
passion for innovative design and performance-driven footwear. With a vision to revolutionize the way people \
experience footwear, Emily and Alex set out to create a brand that prioritizes both fashion-forward aesthetics and \
cutting-edge technology.

At SoleStride, our focus is on empowering individuals to conquer their day with confidence and style. We believe that \
every step should be a stride towards greatness, which is why we meticulously design our shoes to provide unparalleled \
comfort and support, whether you're hitting the pavement for a morning run or striding through the urban jungle.

Introducing the "BestRun" Performance Running Shoe by SoleStride, the latest innovation in our lineup designed for \
serious runners and fitness enthusiasts. The BestRun shoes feature advanced cushioning technology that provides \
exceptional shock absorption, reducing impact on joints and ensuring a smooth, comfortable run."""

    user_id = f"test_customer_support{formatted_time}"
    document_ids = ["ss_prods"]

    cs_agent = VoiceCustomerSupportAgent()
    pc_instance = PineconeIndex()
    local_path = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "SoleStride_products.pdf"
    )

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_file_to_index_pinecone(
            local_path=cls.local_path,
            user_id=cls.user_id,
            document_id=cls.document_ids[0],
            cache_q=False,
        )

        while True:
            time.sleep(2)
            info_dict = cls.pc_instance.freshness()
            list_namespaces = list(info_dict["namespaces"].keys())
            if cls.user_id in list_namespaces:
                break

    def test_00_inprompt_question(self):
        chat_history = ChatHistory(
            [Dialogue("Hello", "Hello, how can I help you?$eos$")]
        )

        question = "Who are you? tell me your name"

        agent_response = self.cs_agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
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
        self.assertIn(agent_response['metadata']['chain'], "inprompt_chain")
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)

    def test_01_master_chain_question(self):
        chat_history = ChatHistory(
            [
                Dialogue("Hello how are you?", "I'm good, thank you for asking!$eos$ How can I assist you today?$eos$"),
                Dialogue("What is your name?", "I am an AI Agent, your helpful customer support assistant.$eos$ How can I help you today?$eos$"),
            ]
        )

        question = "Could you tell me about other products like `urban flex`?"

        agent_response = self.cs_agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join([token if type(token) is str else "" for token in agent_response["response"]])

        self.assertIsInstance(full_response, str)
        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertIsInstance(agent_response['metadata'], dict)
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertIsInstance(agent_response['metadata']['references'][0]['text'], str)
        self.assertIn(agent_response['metadata']['chain'], "master_chain")
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.user_id)


if __name__ == "__main__":
    unittest.main()
