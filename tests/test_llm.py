from mtn_agents.llm import OpenAILLM, query_stream_sentence

from typing import Generator, List
import unittest


class TestOpenAILLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_instance = OpenAILLM()

    def test_00_openai_init(self):
        self.assertIsNotNone(self.llm_instance)

    def test_01_openai_query(self):
        self.assertIsInstance(self.llm_instance.query("Hello"), str)
        self.assertGreater(self.llm_instance.prompt_tokens, 0)
        self.assertGreater(self.llm_instance.completion_tokens, 0)

    def test_02_openai_query_stream(self):
        self.assertIsInstance(self.llm_instance.query_stream("Hello"), Generator)

    def test_03_openai_embedding(self):
        self.assertIsInstance(self.llm_instance.embedding(["hello"])[0], List)

    def test_04_openai_query_stream_sentence(self):
        word_by_word = self.llm_instance.query_stream("Hello")
        sent_by_sent = query_stream_sentence(text=word_by_word)

        self.assertIsInstance(sent_by_sent, Generator)


# class TestAnthropicLLM(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.llm_instance = AnthropicLLM()
#
#     def test_00_openai_init(self):
#         self.assertIsNotNone(self.llm_instance)
#
#     def test_01_openai_query(self):
#         self.assertIsInstance(self.llm_instance.query("Hello"), str)
#
#     def test_02_openai_query_stream(self):
#         self.assertIsInstance(self.llm_instance.query_stream("Hello"), Generator)


if __name__ == "__main__":
    unittest.main()
