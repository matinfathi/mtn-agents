from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.utils import ChatHistory, Dialogue, Presentation
from mtn_agents.retriever import PineconeIndex
from mtn_agents.agents import TeachAgent

from types import GeneratorType
from datetime import datetime
import unittest
import pathlib
import time
import os


class TestTeachAgent(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime("%B_%d_%Y_%H:%M:%S")

    product_description = dict(
        Presentation(
            [
                """Machine Learning (ML) is a subset of artificial intelligence that focuses on developing algorithms that \
allow computers to learn from and make decisions based on data. By analyzing patterns and relationships within large \
datasets, ML models can improve their performance over time without being explicitly programmed for specific tasks. \
This technology is widely used across various industries, including finance, healthcare, and technology, to automate \
processes, predict trends, and gain insights from data.""",
                """Machine Learning can be categorized into three main types: supervised, unsupervised, and reinforcement \
learning. In supervised learning, models are trained on labeled data, learning to predict outcomes based on \
input-output pairs. Unsupervised learning, on the other hand, deals with unlabeled data, where the model identifies \
hidden patterns or groupings within the data. Reinforcement learning involves training models through a system of \
rewards and penalties, optimizing decision-making processes over time.""",
                """Machine Learning has a wide range of applications. In healthcare, ML models help in diagnosing diseases, \
personalizing treatments, and predicting patient outcomes. In finance, ML is used for fraud detection, algorithmic \
trading, and credit scoring. In marketing, it powers personalized recommendations, customer segmentation, and \
sentiment analysis. As data availability and computational power increase, the potential for ML to revolutionize \
industries continues to grow.""",
                """Deep Learning is a specialized subset of Machine Learning that utilizes neural networks with multiple \
layers (hence "deep") to model complex patterns in data. Inspired by the human brain's structure, these deep \
neural networks are particularly effective in processing and analyzing large amounts of unstructured data, such as \
images, audio, and text. Deep Learning has achieved significant breakthroughs in fields like computer vision, natural \
language processing, and speech recognition.""",
                """The impact of Deep Learning is evident in many cutting-edge technologies. In computer vision, deep learning \
models power facial recognition, autonomous vehicles, and medical image analysis. In natural language processing, they \
enable advanced chatbots, language translation, and sentiment analysis. As Deep Learning models continue to evolve, \
they push the boundaries of what machines can achieve, opening new possibilities in AI-driven innovation.""",
            ]
        )
    )

    user_id = f"test_teach_agent_{formatted_time}"
    document_ids = ["ml_doc"]

    agent = TeachAgent()
    pc_instance = PineconeIndex()

    local_path = os.path.abspath(pathlib.Path(__file__).parent / "files" / "ML_DL.pdf")

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_file_to_index_pinecone(
            local_path=cls.local_path,
            user_id=cls.user_id,
            document_id=cls.document_ids[0],
            cache_q=False,
            is_slide=True,
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

        full_response = "".join([token if type(token) is str else "" for token in agent_response["response"]])

        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertIn(agent_response["metadata"]["slide_id"].lower(), ["none", "page_1"])
        self.assertEqual(agent_response["metadata"]["chain"], "inprompt_chain")
        self.assertIsInstance(agent_response["metadata"]["prompt"], str)
        self.assertGreater(len(agent_response["metadata"]["prompt"]), 10)
        self.assertEqual(agent_response["metadata"]["document_id"], "current")

    def test_01_master_chain_question(self):
        chat_history = ChatHistory(
            [Dialogue(user="", assistant="Hello!$eos$ This is mtn_agents.$eos$ How are you today?$eos$")]
        )

        question = "Could you explain other concepts than machine learning like deep learning?"

        agent_response = self.agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="mtn_agents",
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join([token if type(token) is str else "" for token in agent_response["response"]])

        self.assertIsInstance(full_response, str)
        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertEqual(agent_response["metadata"]["slide_id"].lower(), "page_4")
        self.assertIn(agent_response["metadata"]["chain"], ["master_chain", "inprompt_chain"])
        self.assertIsInstance(agent_response["metadata"]["prompt"], str)
        self.assertGreater(len(agent_response["metadata"]["prompt"]), 10)

        self.assertIsInstance(agent_response["metadata"]["references"], list)
        self.assertIsInstance(agent_response["metadata"]["references"][0]["text"], str)
        self.assertEqual(agent_response["metadata"]["document_id"], self.document_ids[0])

    def test_02_inprompt_chain_question(self):
        chat_history = ChatHistory(
            [Dialogue(user="", assistant="Hello!$eos$ This is mtn_agents.$eos$ How are you today?$eos$")]
        )

        question = "[continue presentation]"

        agent_response = self.agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="mtn_agents",
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join([token if type(token) is str else "" for token in agent_response["response"]])

        self.assertIsInstance(full_response, str)
        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertEqual(agent_response["metadata"]["chain"], "inprompt_chain")
        self.assertIsInstance(agent_response["metadata"]["prompt"], str)
        self.assertGreater(len(agent_response["metadata"]["prompt"]), 10)
        self.assertEqual(agent_response["metadata"]["document_id"], "current")

    def test_03_end_presentation(self):
        chat_history = ChatHistory(
            [Dialogue(user="", assistant="Hello!$eos$ This is mtn_agents.$eos$ How are you today?$eos$")]
        )

        question = "[continue presentation]"

        agent_response = self.agent(
            product_description=self.product_description,
            question=question,
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="mtn_agents",
        )

        self.assertIsInstance(agent_response, dict)

        full_response = "".join([token if type(token) is str else "" for token in agent_response["response"]])

        self.assertIsInstance(full_response, str)
        self.assertTrue(full_response.strip().endswith("$eos$"))
        self.assertEqual(agent_response["metadata"]["chain"], "inprompt_chain")
        self.assertIsInstance(agent_response["metadata"]["prompt"], str)
        self.assertGreater(len(agent_response["metadata"]["prompt"]), 10)
        self.assertEqual(agent_response["metadata"]["document_id"], "current")

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.user_id)


if __name__ == "__main__":
    unittest.main()
