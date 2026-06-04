from mtn_agents.features import (
    summarize_text,
    ranking,
    mckw_en,
    get_followup_question,
    get_followup_question_v2,
    get_conv_insights,
    get_presentation_feedback,
)
from mtn_agents.utils import (
    Language,
    ChatHistory,
    Dialogue,
    Presentation,
)

import unittest
import pathlib
import os


class TestFeatures(unittest.TestCase):
    product_description = Presentation([
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
        """Presenting the 'SkyHigh' Performance Trainer: Engineered for athletes and fitness \
enthusiasts, our SkyHigh trainers offer unparalleled performance and style. With a lightweight yet durable design, \
these shoes provide exceptional support and stability during high-intensity workouts. The SkyHigh trainers feature \
responsive cushioning and a secure lace-up closure, ensuring a comfortable and personalized fit for every wearer.""",
        """Introducing the 'CityStride' Casual Shoe: Perfect for everyday wear and casual outings, our \
CityStride shoes blend urban style with all-day comfort. Featuring a versatile design and premium materials, these \
shoes are ideal for exploring city streets or running errands. The CityStride shoes boast a cushioned insole and \
breathable lining, keeping your feet feeling fresh and comfortable from morning till night.""",
    ])
    local_path_en = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "features_doc_en.txt"
    )
    local_path_ar = os.path.abspath(
        pathlib.Path(__file__).parent / "files" / "features_doc_ar.txt"
    )
    text_en = ""
    text_ar = ""

    @classmethod
    def setUpClass(cls):
        with open(cls.local_path_en, "r", encoding="utf8") as f:
            cls.text_en = f.read()
        with open(cls.local_path_ar, "r", encoding="utf8") as f:
            cls.text_ar = f.read()

    def test_00_suggest_question(self):
        sq = get_followup_question(
            text=self.text_en, top_k=3, language=Language.ENGLISH
        )

        self.assertIsInstance(sq, list)
        self.assertGreater(len(sq), 0)
        self.assertTrue(bool(sq[0]["question"]))
        self.assertTrue(bool(sq[0]["answer"]))

    def test_00_suggest_question_v2(self):
        chat_history = ChatHistory(
            [
                Dialogue(
                    user="",
                    assistant="Hi there! I'm Homayoon, and I'm excited to share with you about SoleStride, "
                    "a dynamic footwear company. How's your day going?",
                    slide_id="None",
                ),
                Dialogue(
                    user="I'm fine thanks",
                    assistant="That's great to hear! SoleStride is all about combining style and comfort in our "
                    "footwear. Are you currently looking for any specific type of shoes?",
                    slide_id="page_2",
                ),
            ]
        )
        sq = get_followup_question_v2(
            product_description=dict(self.product_description),
            chat_history=chat_history,
            response_metadata={
                "intent": "None",
                "references": [
                    {
                        "score": 0.1,
                        "text": "This is the reference text.",
                        "document_id": "Document",
                    }
                ],
                "slide_id": "None",
            }
        )

        self.assertIsInstance(sq, list)
        self.assertGreater(len(sq), 0)
        self.assertTrue(bool(sq[0]["question"]))
        self.assertTrue(bool(sq[0]["answer"]))

    def test_01_ranking(self):
        ref_list = ["It's good day!", "goodbye my friend.", "hello, how are you?"]
        text = "hello nice to meet you."
        new_ref_list = ranking(answer=text, ref_list=ref_list)

        self.assertEqual(new_ref_list[0], ref_list[-1])

    def test_02_summarize(self):
        # len_summary = int(len(self.text_en) / 4)
        summary = summarize_text(
            text=self.text_en, summarize_tokens=50, lang=Language.ENGLISH
        )

        self.assertIs(type(summary), str)
        self.assertGreater(len(self.text_en), len(summary))

    def test_03_mckw(self):
        list_mckw = mckw_en(target_text=self.text_en)

        self.assertIs(type(list_mckw), list)

    def test_04_suggest_question_ar(self):
        sq = get_followup_question(text=self.text_ar, top_k=3, language=Language.ARABIC)

        self.assertIsInstance(sq, list)
        self.assertGreater(len(sq), 0)
        self.assertTrue(bool(sq[0]["question"]))
        self.assertTrue(bool(sq[0]["answer"]))

    def test_05_ranking_ar(self):
        ref_list = ["إنه يوم جيد!", "وداعًا يا صديقي.", "مرحبًا، كيف حالك؟"]
        text = "مرحبًا، سعيد بلقائك."
        new_ref_list = ranking(answer=text, ref_list=ref_list)

        self.assertEqual(new_ref_list[0], ref_list[-1])

    def test_06_summarize_ar(self):
        # len_summary = int(len(self.text_en) / 4)
        summary = summarize_text(
            text=self.text_ar, summarize_tokens=50, lang=Language.ARABIC
        )

        self.assertIs(type(summary), str)
        self.assertGreater(len(self.text_en), len(summary))

    def test_07_mckw_ar(self):
        list_mckw = mckw_en(target_text=self.text_en)

        self.assertIs(type(list_mckw), list)

    def test_08_conv_insights(self):
        chat_history = ChatHistory(
            [
                Dialogue(
                    "",
                    "Hi there! This is mtn_agents from L'Oreal USA S/D, Inc. How are you today?",
                ),
                Dialogue(
                    "Hi, good thank you",
                    "Great to hear that you're doing well! I'd love to introduce you to one of our popular products: "
                    "**CeraVe Hydrating Facial Cleanser**. It's gentle, fragrance-free, and suitable for all skin "
                    "types. Are there any specific needs or concerns you have with your current skincare routine?",
                ),
                Dialogue(
                    "Yeah I have some skin dryness and most of the times the moisturizers don't help.",
                    "CeraVe Hydrating Facial Cleanser could be a great addition to your routine! It's designed to be gentle "
                    "yet effective at removing dirt and makeup without stripping your skin's natural oils. It also contains "
                    "hyaluronic acid to attract moisture and ceramides to help restore your skin's natural barrier. This "
                    "combination can help address your dryness issues. Would you like to know where you can purchase it or "
                    "have other questions?",
                ),
                Dialogue(
                    "Sure, how can I purchase it and how much is it?",
                    "You can purchase CeraVe Hydrating Facial Cleanser from a variety of places. Here are some options: * "
                    "**Drugstores and pharmacies** like CVS or Walgreens. * **Online retailers** such as Amazon, "
                    "Lookfantastic, or the CeraVe website itself. * **Mass retailers** like Walmart or Target. Prices "
                    "generally range between €10.29 and €11.95 depending on the retailer and size (236ml or 473ml). Would "
                    "you like help finding the best deal or have any other questions?",
                ),
                Dialogue(
                    "Yes do you know anything about the discounts?",
                    "Actually you can find this information on the website.",
                ),
            ]
        )

        response = get_conv_insights(chat_history)

        self.assertIs(type(response), dict)
        self.assertIn("conversation_summary", list(response.keys()))
        self.assertIn("conversion_score", list(response.keys()))
        self.assertIn("key_topics", list(response.keys()))
        self.assertIn("open_queries", list(response.keys()))
        self.assertIs(type(response["conversation_summary"]), str)
        self.assertIs(type(response["conversion_score"]), int)
        self.assertIs(type(response["key_topics"]), list)
        self.assertIs(type(response["open_queries"]), list)
        self.assertIn(response["conversion_score"], list(range(11)))
        self.assertGreater(len(response["key_topics"]), 0)
        self.assertGreater(len(response["open_queries"]), 0)

    def test_09_feedback(self):
        feedback = get_presentation_feedback(product_description=self.product_description)

        self.assertIs(type(feedback), str)


if __name__ == "__main__":
    unittest.main()
