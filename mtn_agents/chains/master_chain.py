from typing import List, Dict, Optional, Any, Union
from ast import literal_eval
import logging

from ..constants import MasterChainConfig, LATEST_EMBEDDING_MODEL
from .utils import create_standalone_question, prepare_filters_for_reference
from ..features.embedding_ranking import ranking_pinecone
from ..prompt_templates import prompt_master_chain
from ..utils import Language, ChatHistory
from ..retriever import BaseIndex
from ..llm import BaseLLM
from .base import BaseChain

logger = logging.getLogger(__name__)


class MasterChain(BaseChain):
    def __init__(self, generator: Optional[BaseLLM] = None, retriever: Optional[BaseIndex] = None, **kwargs):
        super().__init__(generator=generator, retriever=retriever, **kwargs)

    def __call__(
        self,
        question: str,
        user_id: str,
        document_ids: List[str],
        chat_history: Optional[Union[ChatHistory, None]] = None,
        stream: Optional[bool] = False,
        ranking: Optional[bool] = False,
        language: Optional[Language] = Language.ENGLISH,
        top_k: Optional[int] = MasterChainConfig.MASTER_CHAIN_TOP_K_REFERENCE,
        threshold: Optional[float] = MasterChainConfig.MASTER_CHAIN_THRESHOLD,
        standalone: Optional[bool] = True,
        agent_name: Optional[str] = "Jamie",
        embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
        agent_uuid: Optional[str] = "",
        main_document_id: Optional[str] = "",
    ) -> Dict[str, Any]:
        """Processes a question-and-answer (QA) interaction with an embedding index and LLM."""
        logger.info("Using Master Chain.")
        filters = prepare_filters_for_reference(document_ids, embedding_model, agent_uuid, main_document_id)

        if chat_history:
            if standalone:
                standalone_question = create_standalone_question(
                    llm=self.generator, chat_history=chat_history.to_str("AI Agent"), question=question
                )
                logger.info(
                    (
                        f"\n----------------------\nStand alone question is --> {standalone_question}\n"
                        "----------------------\n"
                    )
                )

                pc_ref_list = self.retriever.detailed_query(
                    embedding_model=embedding_model,
                    input_texts=[standalone_question],
                    user_id=user_id,
                    filter_dict=filters,
                    top_k=top_k,
                    include_values=True,
                )["matches"]

            else:
                pc_ref_list = self.retriever.detailed_query(
                    embedding_model=embedding_model,
                    input_texts=[chat_history.to_str(assistant_name="AI Agent") + question],
                    user_id=user_id,
                    filter_dict=filters,
                    top_k=top_k,
                    include_values=True,
                )["matches"]
        else:
            pc_ref_list = self.retriever.detailed_query(
                embedding_model=embedding_model,
                input_texts=[question],
                user_id=user_id,
                filter_dict=filters,
                top_k=top_k,
                include_values=True,
            )["matches"]

        ref_list = []
        tools = []

        for item in pc_ref_list:
            if item["score"] >= threshold:
                if item["metadata"]["chunk_type"] == "tools":
                    tools.extend(literal_eval(item["metadata"]["tools"]))
                    continue

                ref_list.append(item["metadata"]["text"])

        required_text = {
            "references": "\n".join(ref_list),
            "language": language.description,
            "agent_name": agent_name,
        }

        generated_answer_dict = super().__call__(
            question=question,
            chat_history_str=chat_history.to_str(assistant_name=agent_name) if chat_history else "",
            required_texts=required_text,
            stream=stream,
            prompt=prompt_master_chain,
            tools=tools if tools else None,
        )

        if generated_answer_dict["response_type"] == "function_call":
            generated_answer_dict["metadata"]["chain"] = "master_chain"
            return generated_answer_dict

        if ranking and ref_list and not stream:
            pc_ref_list = ranking_pinecone(text=generated_answer_dict["response"], ref_list=pc_ref_list)

        new_pc_list = []

        for record in pc_ref_list:
            new_pc_dict = {key: value for key, value in record.to_dict().items() if key not in ["values", "metadata"]}
            new_pc_dict.update(record.get("metadata", {}))
            new_pc_list.append(new_pc_dict)

        response = {
            "response": generated_answer_dict["response"],
            "response_type": "response",
            "metadata": {
                "prompt_tokens": self.generator.prompt_tokens,
                "completion_tokens": self.generator.completion_tokens,
                "references": new_pc_list,
                "prompt": generated_answer_dict["metadata"]["prompt"],
                "chain": "master_chain",
            },
        }

        return response
