import pathlib
import os

# File path
GREET_PICKLE_PATH = os.path.abspath(
    pathlib.Path(__file__).parent / "files" / "greet_pickle_ada.pkl"
)
MODEL_PICKLE_PATH = os.path.abspath(
    pathlib.Path(__file__).parent / "files" / "model_pickle_ada.pkl"
)
MTN_AGENTS_PICKLE_PATH = os.path.abspath(
    pathlib.Path(__file__).parent / "files" / "mtn_agents_pickle_ada.pkl"
)

GREET_AR_PICKLE_PATH = os.path.abspath(
    pathlib.Path(__file__).parent / "files" / "greet_ar_pickle_ada.pkl"
)
MODEL_AR_PICKLE_PATH = os.path.abspath(
    pathlib.Path(__file__).parent / "files" / "model_ar_pickle_ada.pkl"
)
MTN_AGENTS_AR_PICKLE_PATH = os.path.abspath(
    pathlib.Path(__file__).parent / "files" / "mtn_agents_ar_pickle_ada.pkl"
)


# RAGV1 constants
class MasterChainConfig:
    MASTER_CHAIN_THRESHOLD = 0.27
    MASTER_CHAIN_TOP_K_REFERENCE = 10


# RAGV2 constants
class RagV2ConfigEn:
    RAGV2_CACHE_THRESHOLD_UP = 0.893
    RAGV2_CACHE_THRESHOLD_DOWN = 0.84
    RAGV2_DEFAULT_CACHE_LIST = ["company", "greet", "model"]
    RAGV2_TOP_K_REFERENCES = 5
    RAGV2_TOP_K_CACHES = 1


class RagV2ConfigAr:
    RAGV2_DEFAULT_CACHE_LIST = ["company_ar", "greet_ar", "model_ar"]
    RAGV2_CACHE_THRESHOLD_UP = 0.89
    RAGV2_CACHE_THRESHOLD_DOWN = 0.89
    RAGV2_TOP_K_REFERENCES = 5
    RAGV2_TOP_K_CACHES = 1


class MasterChainVisualConfig:
    MASTER_CHAIN_THRESHOLD = 0.27
    MASTER_CHAIN_TOP_K_REFERENCE = 10


# OpenAI constants
OPENAI_REASONING_MODEL_O1 = "o4-mini"
OPENAI_CHAT_MODEL_GPT_4O = "gpt-4o"
OPENAI_CHAT_MODEL_GPT_4O_MINI = "gpt-4o-mini"
OPENAI_CHAT_MODEL_GPT_35 = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL_2 = "text-embedding-ada-002"
OPENAI_EMBEDDING_MODEL_3 = "text-embedding-3-small"
JSON_RESPONSE_FORMAT = {"type": "json_object"}

# Anthropic constants
ANTHROPIC_CHAT_MODEL_SONNET_35 = "claude-3-5-sonnet-20240620"
ANTHROPIC_CHAT_MODEL_MAX_TOKEN = 1024

# Features' constants
# Summarize
CONTEXT_WINDOW = 4000
SUMMARIZE_LENGTH = 200
AUTO_SLIDE_GENERATION_RETRIEVE_THRESHOLD = 0.25


# Action Items
class ActionItems:
    HEADER = """\n\n**Call to Action List:**\nThis is a list of actions the user can take. The agent should inform the \
user about these actions when asked."""
    WEBSITE_RULE = """- Users can visit the product's website. Guide them to do so at the end of the conversation. \
e.g. {{"1_used_reference_id": "None", "2_text": "For more information and to make a purchase, you can visit the \
[company_name] website.$eos$"}}"""
    MEET_RULE = """- Users can schedule a meeting with a real agent. Direct them to use the calendar for scheduling. \
e.g. {{"1_used_reference_id": "None", "2_text": "Would you like to schedule a meeting?$eos$ If so, you can use the \
calendar in the corner of the screen.$eos$"}}"""


CONSECUTIVE_PRESENT_QUESTION = "[{agent_name} should continue the presentation and move to next slide]"
CONVERSATION_START_QUESTION = "[{agent_name} should start the conversation with a brief greeting and await a response]"
CONVERSATION_START_QUESTION_TEACH_AGENT = "[{agent_name} should start the conversation with a brief greeting]"

LATEST_EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL_3
