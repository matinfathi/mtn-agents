from mtn_agents.agents import VisualSalesAgent
from mtn_agents.upload_reference import upload_file_to_index_pinecone
from mtn_agents.retriever import PineconeIndex
from mtn_agents.utils import (
    ChatHistory,
    Dialogue,
    Presentation,
)

from types import GeneratorType
from datetime import datetime
import unittest
import pathlib
import time
import os


class TestVisualSalesAgent(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime("%B_%d_%Y_%H:%M:%S")

    product_description = dict(Presentation([
        """Introducing SoleStride, a dynamic footwear company dedicated to crafting premium shoes that \
marry style with comfort. Founded by avid runners and footwear enthusiasts Emily Parker and Alex Chen, SoleStride was \
born out of a shared passion for innovative design and performance-driven footwear. With a vision to revolutionize the \
way people experience footwear, Emily and Alex set out to create a brand that prioritizes both fashion-forward \
aesthetics and cutting-edge technology.""",
        """At SoleStride, our focus is on empowering individuals to conquer their day with confidence \
and style. We believe that every step should be a stride towards greatness, which is why we meticulously design \
our shoes to provide unparalleled comfort and support, whether you're hitting the pavement for a morning run or \
striding through the urban jungle.""",
        """Our product lineup features a diverse range of footwear tailored to meet the needs of every \
lifestyle. From sleek sneakers engineered for urban exploration to rugged trail runners designed for off-road \
adventures, each pair of SoleStride shoes is crafted with precision and attention to detail. Our innovative \
materials and construction techniques ensure durability, flexibility, and optimal performance, no matter the \
terrain.""",
        """At SoleStride, we're not just selling shoes; we're offering a lifestyle. Join us on our \
journey as we continue to push the boundaries of footwear innovation and inspire individuals to stride boldly \
towards their dreams. Experience the difference with SoleStride – where style meets comfort, and every step is a \
statement.""",
        """Introducing the 'Urban Flex' Sneaker: Engineered for the modern city dweller, our Urban Flex \
sneakers combine sleek style with unmatched comfort. Featuring a lightweight and breathable design, these sneakers are \
perfect for navigating bustling streets and city commutes. With cushioned soles and a flexible construction, the Urban \
Flex sneakers offer all-day comfort without compromising on style.""",
        """Discover the 'TrailBlaze' Runner: Designed for outdoor enthusiasts and adventure seekers, \
our TrailBlaze runners are built to conquer any terrain. With rugged outsoles and durable materials, these shoes \
provide exceptional grip and stability on rugged trails and uneven surfaces. The TrailBlaze runners feature \
water-resistant uppers and reinforced toe caps, ensuring protection and performance in the great outdoors.""",
        """Introducing the 'ActiveStride' Walking Shoe: Ideal for fitness enthusiasts and everyday \
walkers, our ActiveStride shoes are engineered for maximum comfort and support. Featuring a cushioned midsole and a \
contoured footbed, these shoes provide optimal shock absorption and arch support, reducing fatigue and discomfort \
during long walks or workouts. With a stylish design and breathable mesh uppers, the ActiveStride walking shoes are \
perfect for staying active and comfortable all day long.""",
    ]))

    user_id = f"test_visual_sales_{formatted_time}"
    document_ids = ["sole_stride_ref"]

    vs_agent = VisualSalesAgent()
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

    def test_00_start_conversation(self):
        agent_response = self.vs_agent(
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
        self.assertIn(agent_response['metadata']['slide_id'].lower(), ["none", "page_1"])
        self.assertEqual(agent_response['metadata']['chain'], "inprompt_chain")
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)
        self.assertEqual(agent_response['metadata']['document_id'], "current")

    def test_01_inprompt_chain_question(self):
        chat_history = ChatHistory(
            [
                Dialogue(
                    user="",
                    assistant="Hello!$eos$ This is mtn_agents from SoleStride.$eos$ How are you today?$eos$",
                    assistant_metadata={"slide_id": "page_1", "chain": "inprompt_chain"}
                ),
                Dialogue(
                    user="Hello, Thanks I'm fine",
                    assistant="Great to hear that!$eos$ I'm excited to share a bit about one of our latest products "
                              "that might be a perfect fit for you.$eos$ It's our Urban Flex Sneaker.$eos$ Before I "
                              "dive into the details, may I ask if you have any specific needs or concerns with your "
                              "current footwear?$eos$",
                    assistant_metadata={"slide_id": "page_5", "chain": "inprompt_chain"},
                ),
            ]
        )

        question = "Well I need shoes, could you explain more about urban flex?"

        agent_response = self.vs_agent(
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
        # self.assertEqual(agent_response['metadata']['slide_id'].lower(), "page_5")
        # self.assertIn(agent_response['metadata']['chain'], "inprompt_chain")
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)
        self.assertEqual(agent_response['metadata']['document_id'], "current")

    def test_02_master_chain_question(self):
        chat_history = ChatHistory(
            [
                Dialogue(
                    user="",
                    assistant="Hello!$eos$ This is mtn_agents from SoleStride.$eos$ How are you today?$eos$",
                    slide_id="None",
                ),
                Dialogue(
                    user="Hello, Thanks I'm fine",
                    assistant="Great to hear that!$eos$ I'm excited to share a bit about one of our latest products "
                              "that might be a perfect fit for you.$eos$ It's our Urban Flex Sneaker.$eos$ Before I "
                              "dive into the details, may I ask if you have any specific needs or concerns with your "
                              "current footwear?$eos$",
                    slide_id="page_5",
                ),
            ]
        )

        question = "tell me about other products like `HikerPro`."

        agent_response = self.vs_agent(
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

        full_response = "".join([token if type(token) is str else "" for token in agent_response["response"]])

        self.assertTrue(full_response.endswith("$eos$"))
        self.assertIsInstance(agent_response['metadata'], dict)
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertIn(agent_response['metadata']['chain'], "master_chain")
        self.assertIsInstance(agent_response['metadata']['prompt'], str)
        self.assertGreater(len(agent_response['metadata']['prompt']), 10)
        self.assertIsInstance(agent_response['metadata']['references'], list)
        self.assertIsInstance(agent_response['metadata']['references'][0]['text'], str)
        self.assertEqual(agent_response['metadata']['document_id'], "None")

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.user_id)


if __name__ == "__main__":
    unittest.main()
