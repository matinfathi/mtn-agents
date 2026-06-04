from mtn_agents.utils import (
    ChatHistory,
    Dialogue,
)
from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.agents import VideoAgent

from types import GeneratorType
from datetime import datetime
import unittest
import pathlib
import time
import os


class TestVideoAgent(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime("%B_%d_%Y_%H:%M:%S")

    product_description = """\
SoleStride is a footwear brand founded by Emily Parker and Alex Chen, dedicated to blending style with comfort through \
innovative design. Their shoes, crafted for both performance and aesthetics, cater to various lifestyles, from urban \
exploration to trail running. The brand’s latest product, the "BestRun" Performance Running Shoe, features advanced \
cushioning, a breathable mesh upper, and a responsive midsole for optimal support and shock absorption. Designed for \
serious runners, the BestRun offers durability, traction, and seamless comfort, ensuring users don’t have to \
compromise on either style or functionality."""

    video_transcript = """\
Hey everyone! John here, and today I'm super excited to talk to you about the latest innovation from SoleStride – the \
BestRun Performance Running Shoe. I've been testing these out for a couple of weeks now, and let me tell you, they're \
a game-changer. Whether you're hitting the road or the trails, the durable rubber outsole provides excellent traction. \
The seamless construction also minimizes irritation, so you can stay focused on your run."""

    user_id = f"test_video_agent_{formatted_time}"
    document_ids = ["sole_stride_ref"]

    va_agent = VideoAgent()
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

    def test_00_inprompt(self):
        chat_history = ChatHistory(
            [Dialogue("Hello", "Hello, how can I help you?$eos$")]
        )

        question = "Who are you? tell me your name"

        agent_response = self.va_agent(
            product_description=self.product_description,
            video_transcription=self.video_transcript,
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
        self.assertIn(agent_response['metadata']['chain'], ["master_chain", "inprompt_chain"])
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)

    def test_02_master_chain_question(self):
        chat_history = ChatHistory(
            [
                Dialogue("Hello how are you?", "I'm good, thank you for asking!$eos$ How can I assist you today?$eos$"),
                Dialogue("What is your name?", "I am mtn_agents, your helpful customer support assistant.$eos$ How can I help you today?$eos$"),
            ]
        )

        question = "Could you tell me about other products. for example urban flex shoes."

        agent_response = self.va_agent(
            product_description=self.product_description,
            video_transcription=self.video_transcript,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join(
            [token if type(token) is str else "" for token in agent_response["response"]]
        )

        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertIsInstance(agent_response['metadata'], dict)
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertIsInstance(agent_response['metadata']['references'][0], dict)
        self.assertIsInstance(agent_response['metadata']['references'][0]['text'], str)
        self.assertIn(agent_response['metadata']['chain'], "master_chain")
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.user_id)


if __name__ == "__main__":
    unittest.main()
