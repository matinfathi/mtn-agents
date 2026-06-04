from typing import List, Dict, Optional, Any
from ast import literal_eval
import logging
import json

from ..constants import MasterChainVisualConfig, LATEST_EMBEDDING_MODEL, JSON_RESPONSE_FORMAT
from .utils import create_standalone_question, prepare_filters_for_reference
from ..chains.utils import decode_visual_sales_agent_answer_stream
from ..features.embedding_ranking import ranking_pinecone
from ..prompt_templates import prompt_master_chain_visual
from ..utils import Language, ChatHistory
from ..retriever import BaseIndex
from ..llm import BaseLLM
from .base import BaseChain

logger = logging.getLogger(__name__)


class MasterChainVisual(BaseChain):
    def __init__(self, generator: Optional[BaseLLM] = None, retriever: Optional[BaseIndex] = None, **kwargs):
        super().__init__(generator=generator, retriever=retriever, **kwargs)

    def __call__(
        self,
        question: str,
        user_id: str,
        document_ids: List[str],
        chat_history: Optional[ChatHistory] = None,
        stream: Optional[bool] = False,
        ranking: Optional[bool] = False,
        language: Optional[Language] = Language.ENGLISH,
        top_k: Optional[int] = MasterChainVisualConfig.MASTER_CHAIN_TOP_K_REFERENCE,
        threshold: Optional[float] = MasterChainVisualConfig.MASTER_CHAIN_THRESHOLD,
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
                    llm=self.generator, chat_history=chat_history.to_str("AI Agent", visual=True), question=question
                )
                logger.info(
                    (
                        f"\n----------------------\nStand alone question is --> {standalone_question}\n"
                        f"----------------------\n"
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
        mapping_url = {}
        tools = []

        for idx, item in enumerate(pc_ref_list):
            if item["score"] >= threshold:
                if item["metadata"]["chunk_type"] == "slide":
                    reference_content = item["metadata"]["text"]
                    reference_uuid = (
                        f"""{item["metadata"]["document_id"]}##page_{int(item["metadata"]["page_number"])}"""
                    )
                    assembled_item = (
                        f"""{{"reference_uuid": "{reference_uuid}", "reference_content": "{reference_content}"}}"""
                    )
                    ref_list.append(assembled_item)
                elif item["metadata"]["chunk_type"] == "url":
                    doc_id = item["metadata"]["document_id"]
                    page_id = f"page_{idx + 1}"
                    url_id = item["metadata"].get("source", "No url available")
                    if doc_id not in mapping_url:
                        mapping_url[doc_id] = {}
                    mapping_url[doc_id][page_id] = url_id
                    reference_content = item["metadata"]["text"]
                    reference_uuid = f"""{doc_id}##{page_id}"""
                    assembled_item = (
                        f"""{{"reference_uuid": "{reference_uuid}", "reference_content": "{reference_content}"}}"""
                    )
                    ref_list.append(assembled_item)
                elif item["metadata"]["chunk_type"] == "document":
                    reference_content = item["metadata"]["text"]
                    reference_uuid = f"""document_1##page_{idx + 1}"""
                    assembled_item = (
                        f"""{{"reference_uuid": "{reference_uuid}", "reference_content": "{reference_content}"}}"""
                    )
                    ref_list.append(assembled_item)
                elif item["metadata"]["chunk_type"] == "tools":
                    tools.extend(literal_eval(item["metadata"]["tools"]))

        required_text = {
            "product_description": "\n".join(ref_list),
            "language": language.description,
            "agent_name": agent_name,
        }

        generated_answer_dict = super().__call__(
            question=question,
            chat_history_str=chat_history.to_str(assistant_name=agent_name, visual=True),
            required_texts=required_text,
            stream=stream,
            prompt=prompt_master_chain_visual,
            tools=tools if tools else None,
            response_format=JSON_RESPONSE_FORMAT,
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

        if stream:
            response = decode_visual_sales_agent_answer_stream(generated_answer_dict["response"])
            slide_id = next(response)
        else:
            in_prompt_response = json.loads(generated_answer_dict["response"])
            response = in_prompt_response["2_text"]
            slide_id = in_prompt_response["1_used_reference_id"]

        temp_url = "None"
        slide_id_parts = slide_id.split("##")
        if len(slide_id_parts) > 1:
            document_id, slide_id_temp, *_ = slide_id_parts
            if document_id in list(mapping_url.keys()):
                temp_url = mapping_url[document_id][slide_id_temp]

        response = {
            "response": response,
            "response_type": "response",
            "metadata": {
                "prompt_tokens": self.generator.prompt_tokens,
                "completion_tokens": self.generator.completion_tokens,
                "references": new_pc_list,
                "slide_id": slide_id,
                "chain": "master_chain",
                "prompt": generated_answer_dict["metadata"]["prompt"],
                "url": temp_url,
            },
        }

        return response
