from mtn_agents.llm import OpenAILLM, BaseLLM
from mtn_agents.retriever import PineconeIndex, BaseIndex
from mtn_agents.utils import hex_uuid, Language, read_pickle, ChatHistory
from mtn_agents.features import generate_suggested_questions_with_answer
from mtn_agents.prompt_templates import prompt_chitchat
from mtn_agents.constants import (
    RagV2ConfigEn,
    RagV2ConfigAr,
    GREET_PICKLE_PATH,
    MODEL_PICKLE_PATH,
    MTN_AGENTS_PICKLE_PATH,
    GREET_AR_PICKLE_PATH,
    MODEL_AR_PICKLE_PATH,
    MTN_AGENTS_AR_PICKLE_PATH,
)
from mtn_agents.chains.utils import (
    prepare_filters_for_cache,
    create_standalone_question,
    process_response,
)
from .base import BaseChain
from .master_chain import MasterChain


from typing import List, Dict, Tuple, Optional, Union, Any
import json


def create_route_base_on_description(
    description: str, lang: Language
) -> List[Dict[str, Union[List[str], str]]]:
    data = []
    qas = generate_suggested_questions_with_answer(
        description, num_question=10, language=lang
    )
    qas_string = [str(item) for item in qas]
    qas_embedding = OpenAILLM().embedding(qas_string)
    for idx in range(len(qas_embedding)):
        data.append(
            {
                "id": hex_uuid(),
                "values": qas_embedding[idx],
                "metadata": {
                    "text": qas[idx],
                    "cache_id": "company",
                    "document_id": "company",
                },
            }
        )

    return data


class SemanticRouterChain(BaseChain):
    master_chain: BaseChain = None

    def __init__(self, generator: Optional[BaseLLM] = None, retriever: Optional[BaseIndex] = None, **kwargs):
        super().__init__(generator=generator, retriever=retriever, **kwargs)
        self.master_chain = MasterChain(retriever=self.retriever, generator=self.generator)

    def _semantic_router(
        self,
        text: str,
        namespace: str,
        filters: Dict[str, Dict[str, List[str]]],
        top_k: int = 1,
    ) -> Tuple[str, float, str]:
        response = self.retriever.detailed_query(
            input_texts=[text],
            user_id=namespace,
            filter_dict=filters,
            top_k=top_k,
        )

        if not len(response.get("matches", [])):
            raise Exception(
                "Pinecone returned empty list in semantic router. please first run init_user in RagV2"
            )

        document_id = response["matches"][0]["metadata"].get("document_id", "")
        distance = response["matches"][0]["score"]
        query_text = response["matches"][0]["metadata"].get("text", "")

        intent = document_id

        return intent, distance, query_text

    def _chitchat(
        self,
        question: str,
        stream: Optional[bool] = False,
    ) -> str:
        variables = {
            "question": question
        }
        prompt = prompt_chitchat.text.format(**variables)

        if stream:
            response = self.generator.query_stream(
                prompt=prompt,
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
            )
        else:
            response = self.generator.query(
                prompt=prompt,
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
            )

        return json.loads(response)["response"]

    def __call__(
        self,
        chat_history: Union[ChatHistory, None],
        question: str,
        user_id: str,
        document_ids: List[str],
        cache_list: Optional[List[str]] = None,
        stream: Optional[bool] = False,
        ranking: Optional[bool] = True,
        semantic_router: Optional[bool] = True,
        language: Optional[Language] = Language.ENGLISH,
    ) -> Dict[str, Any]:
        """
        Processes a question-and-answer (QA) interaction, applying semantic routing and cache mechanisms.

        This method enhances QA interactions by incorporating a semantic routing layer that decides whether to fetch
        answers from a cache or proceed with a regular QA process. It takes into account the chat history, the user's
        question, and contextual information to provide a tailored response.

        Args:
            - chat_history (ChatHistory): An instance representing the chat history up to the current question.
            - question (str): The user's question as a string.
            - user_id (str): A string identifier for the user asking the question.
            - document_ids (List[str]): A list of document identifiers relevant to the current context.
            - cache_list (Optional[List[str]]): An optional list of cached responses for efficiency. Defaults to a \
            predefined cache list.
            - stream (Optional[bool]): An optional boolean to determine usage of stream mode or not. if so the ranking \
            feature will be disabled.
            - ranking (Optional[bool]): An optional boolean to determine usage of ranking or not.
            - semantic_router (Optional[bool]): An optional boolean to determine usage of semantic router or not.
            - language (Optional[Language]): The language of the question, the default value is English.

        Returns:
            - Dict[str, Union[str, Dict[str, str]]]: A dictionary with 'response' and 'intent' and \
            'reference_metadata'. 'response' contains the answer to the question, while 'intent' includes The \
            classified intent for the user questions [greet, model, company, reference], 'reference_metadata' is the \
            additional information related to the extracted references for that question.
        """
        rag_config = RagV2ConfigAr if language == Language.ARABIC else RagV2ConfigEn
        cache_list = cache_list or rag_config.RAGV2_DEFAULT_CACHE_LIST

        intent, distance, reference = "", 0, ""

        filters_for_cache = prepare_filters_for_cache(document_ids, cache_list)

        if semantic_router:
            intent, distance, reference = self._semantic_router(
                question, user_id, filters_for_cache, top_k=rag_config.RAGV2_TOP_K_CACHES
            )

        if chat_history and distance < rag_config.RAGV2_CACHE_THRESHOLD_UP:
            chat_history = chat_history.to_str() if type(chat_history) is ChatHistory else chat_history
            question = create_standalone_question(
                llm=self.generator, chat_history=chat_history, question=question
            )

            if semantic_router:
                intent, distance, reference = self._semantic_router(
                    question, user_id, filters_for_cache, top_k=rag_config.RAGV2_TOP_K_CACHES
                )

        if intent in cache_list + document_ids and distance >= rag_config.RAGV2_CACHE_THRESHOLD_UP:
            answer = process_response(reference)

        elif intent in cache_list and distance > rag_config.RAGV2_CACHE_THRESHOLD_DOWN:
            answer = self._chitchat(question, stream)

        else:
            intent = "reference"
            answer = ""

        return {
            "response": answer,
            "metadata": {
                "intent": intent.replace("_ar", ""),
            }
        }

    def init_user(
        self,
        user_id: str,
        vectors: Optional[List[Dict[str, Any]]] = None,
        company_description: Optional[str] = "",
        language: Optional[Language] = Language.SOURCE,
    ):
        if language.code == Language.ARABIC.code:
            pickle_greet = read_pickle(GREET_AR_PICKLE_PATH)
            pickle_model = read_pickle(MODEL_AR_PICKLE_PATH)
        elif language.code == Language.ENGLISH.code:
            pickle_greet = read_pickle(GREET_PICKLE_PATH)
            pickle_model = read_pickle(MODEL_PICKLE_PATH)
        else:
            pickle_greet = read_pickle(GREET_PICKLE_PATH) + read_pickle(
                GREET_AR_PICKLE_PATH
            )
            pickle_model = read_pickle(MODEL_PICKLE_PATH) + read_pickle(
                MODEL_AR_PICKLE_PATH
            )

        if company_description:
            company_ = create_route_base_on_description(
                company_description, language.description
            )
        else:
            if language.code == Language.ARABIC.code:
                company_ = read_pickle(MTN_AGENTS_AR_PICKLE_PATH)
            elif language.code == Language.ENGLISH.code:
                company_ = read_pickle(MTN_AGENTS_PICKLE_PATH)
            else:
                company_ = read_pickle(MTN_AGENTS_PICKLE_PATH) + read_pickle(MTN_AGENTS_AR_PICKLE_PATH)

        if vectors is None:
            vectors = pickle_greet + pickle_model + company_
        else:
            vectors += pickle_greet + pickle_model + company_

        self.retriever.detailed_upsert(vectors, user_id)
