from mtn_agents.upload_reference import upload_tools_to_index_pinecone
from mtn_agents.utils import ChatHistory, Dialogue, Presentation, Slide
from mtn_agents.retriever import PineconeIndex
from mtn_agents.agents import TeachAgent, VoiceCustomerSupportAgent

from datetime import datetime
import unittest
import time


class TestTeachAgent(unittest.TestCase):
    dt_object = datetime.fromtimestamp(time.time())
    formatted_time = dt_object.strftime("%B_%d_%Y_%H:%M:%S")

    product_description = dict(Presentation(
            [
                Slide(
                    slide_number="page_1",
                    slide_content="## slide_content: Transform Your Data Experience with Domo: Let's start by understanding what Domo is. Domo is a cloud-based data platform designed to revolutionize the way businesses handle and utilize data. With Domo, you can create insightful, data-driven strategies through intuitive dashboards and applications. Its user-friendly tools make it easier for businesses to visualize and analyze their data in meaningful ways.",
                ),
                Slide(
                    slide_number="page_2",
                    slide_content="## slide_content: Why Domo: Domo empowers users across your organization, from executives to frontline employees, to work with data effectively. It leverages advanced AI and machine learning for intelligent insights and predictive analytics, helping businesses anticipate trends and make informed decisions. Additionally, Domo is scalable and secure, ensuring your data is compliant with industry standards, providing peace of mind that your data is protected.",
                ),
                Slide(
                    slide_number="page_3",
                    slide_content="## slide_content: Domo Features: Domo empowers everyone with data through Self-Service Analytics, providing drag-and-drop dashboards and user-friendly data exploration. It offers AI-powered recommendations and visualizations for data-driven decisions, seamless integration with popular business applications for a unified data view, and mobile access to stay informed and analyze data on the go. Domo is also your secure, scalable data platform.",
                ),
                Slide(
                    slide_number="page_4",
                    slide_content="## slide_content: Domo Vision & Mission: Domo's vision is a personalized, secure, AI-powered data experience for everyone by offering AI-Driven Insights, Augmented Intelligence, and continuous innovation. Their mission is to empower every user to make informed decisions with data by providing a Centralized Data Platform, Self-Service Analytics, and Actionable Insights, translating complex data into clear, actionable strategies for your business.",
                ),
                Slide(
                    slide_number="page_5",
                    slide_content="## slide_content: Target Markets: Domo caters to various businesses such as Enterprises with complex data needs and dedicated IT teams, Mid-Market businesses looking to centralize data and gain insights with user-friendliness and affordability, and Specific Industries like Financial Services, Retail, and Manufacturing, each having tailored solutions to address specific data challenges.",
                ),
                Slide(
                    slide_number="page_6",
                    slide_content="## slide_content: Domo vs. The Competition: What makes Domo stand out includes its User-Friendly Interface suitable for non-technical users, Flexibility to adapt to existing infrastructure and scale with needs, Seamless Integrations with a wide range of business applications, and Immersive Experiences empowering the creation of impactful and interactive data visualizations.",
                ),
                Slide(
                    slide_number="page_7",
                    slide_content="## slide_content: Pricing Plans: Domo offers flexible pricing plans to suit your needs, including: Free Trial for 30 days with full features and unlimited users, Standard plan for teams with lower data volumes, Enterprise plan for organizations with higher data volumes requiring additional support, and Business Critical plan for the highest levels of security, scale, and data volume handling.",
                ),
                Slide(
                    slide_number="page_8",
                    slide_content="## slide_content: About Domo: Domo is a cloud-native data experiences innovator that puts data to work for everyone. Powered by AI, data science, and a secure data foundation, Domo makes data actionable through user-friendly dashboards and apps. It goes beyond basic data storage by unlocking the potential of your data, driving impactful business results.",
                ),
                Slide(
                    slide_number="page_9",
                    slide_content="## slide_content: Contact Us: Get in touch with us! Here are our contact details: Phone (US): +1 801.899.1000, Phone (UK): +44 203 8791006, Email: info@domo.com, Address: 15 Hatfields, South Bank, London SE1 8DJ UK.",
                ),
                Slide(
                    slide_number="page_10",
                    slide_content="## slide_content: Thank You: Thank you for your time! We hope you see the potential of Domo to transform how your organization utilizes data. Visit our website to learn more and start your free trial today!",
                ),
            ]
        )
    )

    product_description_voice = """\
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

    user_id = f"test_integrated_function_{formatted_time}"
    document_ids = ["tools"]

    agent = TeachAgent()
    voice_agent = VoiceCustomerSupportAgent()
    pc_instance = PineconeIndex()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a given latitude and longitude",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "latitude": {"type": "string", "description": "The latitude of the location"},
                        "longitude": {"type": "string", "description": "The longitude of the location"},
                    },
                    "required": ["latitude", "longitude"],
                },
            },
        }
    ]

    @classmethod
    def setUpClass(cls):
        cls.pinecone_data_file = upload_tools_to_index_pinecone(
            tools=str(cls.tools),
            tools_description="Tools for getting the weather information",
            user_id=cls.user_id,
            document_id=cls.document_ids[0],
        )

        while True:
            time.sleep(2)
            info_dict = cls.pc_instance.freshness()
            list_namespaces = list(info_dict["namespaces"].keys())
            if cls.user_id in list_namespaces:
                break

    def test_00_function_calls_with_tools(self):
        chat_history = ChatHistory(
            [
                Dialogue(user="", assistant="Hello!$eos$ This is Jamie.$eos$ How are you today?$eos$", slide_id="None"),
                Dialogue(user="Im fine thanks", assistant="Great to hear!$eos$ Today, I'm excited to introduce you to Domo.$eos$ Domo is a cloud-based data platform designed to transform how businesses handle and utilize data.$eos$ It helps create insightful, data-driven strategies through intuitive dashboards and applications.$eos$", slide_id="page_1"),
            ]
        )
        
        agent_response = self.agent(
            product_description=self.product_description,
            question="How is the weather in paris?",
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Jamie",
            tools=self.tools,
        )

        self.assertIs(agent_response["response"], None)
        self.assertEqual(agent_response["response_type"], "function_call")
        self.assertIsInstance(agent_response["metadata"]["function_call"], list)
        self.assertEqual(agent_response["metadata"]["function_call"][0]["function"], "get_current_weather")

    def test_01_function_calls_without_tools(self):
        chat_history = ChatHistory(
            [
                Dialogue(user="", assistant="Hello!$eos$ This is Jamie.$eos$ How are you today?$eos$", slide_id="None"),
                Dialogue(user="Im fine thanks", assistant="Great to hear!$eos$ Today, I'm excited to introduce you to Domo.$eos$ Domo is a cloud-based data platform designed to transform how businesses handle and utilize data.$eos$ It helps create insightful, data-driven strategies through intuitive dashboards and applications.$eos$", slide_id="page_1"),
            ]
        )
        
        agent_response = self.agent(
            product_description=self.product_description,
            question="How is the weather in paris?",
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Jamie",
            # tools=self.tools,
        )

        self.assertIs(agent_response["response"], None)
        self.assertEqual(agent_response["response_type"], "function_call")
        self.assertIsInstance(agent_response["metadata"]["function_call"], list)
        self.assertEqual(agent_response["metadata"]["function_call"][0]["function"], "get_current_weather")

    def test_02_function_calls_with_tools_voice(self):
        chat_history = ChatHistory(
            [
                Dialogue(user="", assistant="Hello!$eos$ This is Jamie.$eos$ How are you today?$eos$", slide_id="None"),
                Dialogue(user="Im fine thanks", assistant="Great to hear!$eos$ Today, I'm excited to introduce you to Domo.$eos$ Domo is a cloud-based data platform designed to transform how businesses handle and utilize data.$eos$ It helps create insightful, data-driven strategies through intuitive dashboards and applications.$eos$", slide_id="page_1"),
            ]
        )
        
        agent_response = self.voice_agent(
            product_description=self.product_description_voice,
            question="How is the weather in paris?",
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Jamie",
            tools=self.tools,
        )

        self.assertIs(agent_response["response"], None)
        self.assertEqual(agent_response["response_type"], "function_call")
        self.assertIsInstance(agent_response["metadata"]["function_call"], list)
        self.assertEqual(agent_response["metadata"]["function_call"][0]["function"], "get_current_weather")

    def test_03_function_calls_without_tools_voice(self):
        chat_history = ChatHistory(
            [
                Dialogue(user="", assistant="Hello!$eos$ This is Jamie.$eos$ How are you today?$eos$", slide_id="None"),
                Dialogue(user="Im fine thanks", assistant="Great to hear!$eos$ Today, I'm excited to introduce you to Domo.$eos$ Domo is a cloud-based data platform designed to transform how businesses handle and utilize data.$eos$ It helps create insightful, data-driven strategies through intuitive dashboards and applications.$eos$", slide_id="page_1"),
            ]
        )
        
        agent_response = self.voice_agent(
            product_description=self.product_description_voice,
            question="How is the weather in paris?",
            chat_history=chat_history,
            user_id=self.user_id,
            document_ids=self.document_ids,
            stream=True,
            stream_sentence=True,
            agent_name="Jamie",
            # tools=self.tools,
        )

        self.assertIs(agent_response["response"], None)
        self.assertEqual(agent_response["response_type"], "function_call")
        self.assertIsInstance(agent_response["metadata"]["function_call"], list)
        self.assertEqual(agent_response["metadata"]["function_call"][0]["function"], "get_current_weather")

    @classmethod
    def tearDownClass(cls):
        cls.pc_instance.delete_user_id(user_id=cls.user_id)


if __name__ == "__main__":
    unittest.main()
