from typing import Optional, List
import json

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from ..constants import JSON_RESPONSE_FORMAT, LATEST_EMBEDDING_MODEL
from ..prompt_templates import prompt_voice_sales_agent, prompt_visual_sales_agent_v36
from ..chains.utils import decode_visual_sales_agent_answer_stream
from ..types import AgentResponseDict, ProductDescriptionDict
from ..utils import ChatHistory, Presentation
from ..chains import MasterChainVisual
from ..retriever import BaseIndex
from ..llm import BaseLLM
from .base import BaseAgent, VisualBaseAgent


class VoiceSalesAgent(BaseAgent):
    def __init__(
        self,
        generator: Optional[BaseLLM] = None,
        retriever: Optional[BaseIndex] = None,
        model: Optional[str] = "openai",
        **kwargs,
    ):
        super().__init__(generator=generator, retriever=retriever, model=model, **kwargs)

    def __call__(
        self,
        question: Optional[str] = "",
        chat_history: Optional[ChatHistory] = None,
        product_description: Optional[str] = "",
        additional_information: Optional[str] = "",
        agent_name: Optional[str] = "Jamie",
        user_id: Optional[str] = "",
        document_ids: Optional[List[str]] = None,
        agent_uuid: Optional[str] = "",
        main_document_id: Optional[str] = "",
        stream: Optional[bool] = False,
        stream_sentence: Optional[bool] = False,
        end_sign: Optional[str] = "$eos$",
        trim_end_sign: Optional[bool] = False,
        ranking: Optional[bool] = False,
        standalone: Optional[bool] = True,
        num_chat_history_in_prompt: Optional[int] = 6,
        embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
        tools: Optional[ChatCompletionToolParam] = None,
    ) -> AgentResponseDict:
        """
        Handles the conversation with the agent for sales agent.

            Args:
                - question (Optional[str]): The user's question or input. Defaults to "".
                - chat_history (Optional[ChatHistory]): The chat history object. Defaults to None.
                - product_description (Optional[str]): The product description text. Defaults to "".
                - additional_information (Optional[str]): Additional information text. Defaults to "".
                - agent_name (Optional[str]): The name of the agent to set in prompt. Defaults to "Jamie".
                - user_id (Optional[str]): The ID of the user. Defaults to "".
                - document_ids (Optional[List[str]]): List of document IDs. Defaults to None.
                - agent_uuid (Optional[str]): The UUID of the agent. Defaults to an empty string.
                - main_document_id (Optional[str]): The main document ID. Defaults to an empty string.
                - stream (Optional[bool]): Indicates if streaming mode is enabled. Defaults to False.
                - stream_sentence (Optional[bool]): Indicates if streaming sentences is enabled. Defaults to False.
                - end_sign (Optional[str]): The end sign used in streaming mode. Defaults to "$eos$".
                - trim_end_sign (Optional[bool]): Indicates if the end sign should be trimmed from the response. \
                Defaults to False.

            Returns:
                AgentResponse: The agent's response.
        """
        user_first_message = (
            f"""[{agent_name} should start the conversation with a brief greeting and await a response]"""
        )
        if chat_history:
            chat_history = chat_history[-num_chat_history_in_prompt:]
            if chat_history[0].user == "" or chat_history[0].user == "{}":
                chat_history.add_first_user_message(user_first_message=user_first_message)
            chat_history_str = chat_history.to_str(assistant_name=agent_name)
        else:
            chat_history_str = ""
            question = user_first_message

        required_text = {
            "product_description": product_description,
            "additional_information": additional_information,
            "agent_name": agent_name,
        }

        response = self.inprompt_chain(
            question=question,
            chat_history_str=chat_history_str,
            required_texts=required_text,
            stream=stream,
            prompt=prompt_voice_sales_agent,
            tools=tools,
        )

        if response.get("response_type", "response") == "function_call":
            return response

        if stream:
            response = self.handle_stream(
                response=response,
                chat_history=chat_history,
                question=question,
                user_id=user_id,
                document_ids=document_ids,
                stream=stream,
                stream_sentence=stream_sentence,
                end_sign=end_sign,
                trim_end_sign=trim_end_sign,
                ranking=ranking,
                standalone=standalone,
                agent_name=agent_name,
                embedding_model=embedding_model,
                agent_uuid=agent_uuid,
                main_document_id=main_document_id,
            )

        else:
            response = self.handle_non_stream(
                response=response,
                chat_history=chat_history,
                question=question,
                user_id=user_id,
                document_ids=document_ids,
                stream=stream,
                ranking=ranking,
                standalone=standalone,
                agent_name=agent_name,
                embedding_model=embedding_model,
                agent_uuid=agent_uuid,
                main_document_id=main_document_id,
            )

        if "metadata" in response.keys():
            if "chain" not in response["metadata"].keys():
                response["metadata"].update({"chain": "inprompt_chain"})
        else:
            response["metadata"] = {"chain": "inprompt_chain"}

        return response


class VisualSalesAgent(VisualBaseAgent):
    def __init__(
        self,
        generator: Optional[BaseLLM] = None,
        retriever: Optional[BaseIndex] = None,
        model: Optional[str] = "openai",
        use_master_chain: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(
            generator=generator, retriever=retriever, model=model, use_master_chain=use_master_chain, **kwargs
        )
        self.master_chain = MasterChainVisual()

    def __call__(
        self,
        question: Optional[str] = "",
        chat_history: Optional[ChatHistory] = None,
        product_description: Optional[ProductDescriptionDict] = "",
        additional_information: Optional[str] = "",
        agent_name: Optional[str] = "Jamie",
        user_id: Optional[str] = "",
        document_ids: Optional[List[str]] = None,
        agent_uuid: Optional[str] = "",
        main_document_id: Optional[str] = "",
        stream: Optional[bool] = False,
        stream_sentence: Optional[bool] = False,
        end_sign: Optional[str] = "$eos$",
        trim_end_sign: Optional[bool] = False,
        ranking: Optional[bool] = False,
        standalone: Optional[bool] = True,
        num_chat_history_in_prompt: Optional[int] = 6,
        action_item_website: Optional[bool] = False,
        action_item_meet: Optional[bool] = False,
        embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL,
        tools: Optional[ChatCompletionToolParam] = None,
        **kwargs,
    ) -> AgentResponseDict:
        """
        Processes the given question and context to generate an appropriate response from an agent.

            Args:
                - question (Optional[str]): The question posed by the user. Defaults to an empty string.
                - chat_history (Optional[ChatHistory]): The history of the chat conversation. Defaults to None.
                - product_description (Optional[ProductDescription]): Description of the product. Defaults to
                an empty string.
                - additional_information (Optional[str]): Any additional information that may aid in generating a
                response. Defaults to an empty string.
                - agent_name (Optional[str]): The name of the agent generating the response. Defaults to "Jamie".
                - user_id (Optional[str]): The ID of the user. Defaults to an empty string.
                - document_ids (Optional[List[str]]): List of document IDs that might be relevant to the question.
                Defaults to None.
                - agent_uuid (Optional[str]): The UUID of the agent. Defaults to an empty string.
                - main_document_id (Optional[str]): The main document ID. Defaults to an empty string.
                - stream (Optional[bool]): Whether to stream the response incrementally. Defaults to False.
                - stream_sentence (Optional[bool]): Whether to stream the response sentence by sentence. Defaults to
                False.
                - end_sign (Optional[str]): The sign indicating the end of the response. Defaults to "$eos$".
                - trim_end_sign (Optional[bool]): Whether to trim the end sign from the final response. Defaults to
                False.
                - ranking (Optional[bool]): Whether to rank the generated responses. Defaults to False.
                - standalone (Optional[bool]): Whether the generate standalone question. Defaults to False.
                - num_chat_history_in_prompt (Optional[int]): The number of chat history entries to include in the
                prompt. Defaults to 6.
                **kwargs: Additional keyword arguments.

            Returns:
                AgentResponse: The generated response from the agent.
        """
        presentation_instance_full = Presentation.from_json(product_description)

        (presentation_instance, question, chat_history_str, covered_slides, action_items, chat_history_short) = (
            self.prepare_data(
                input_product_description=presentation_instance_full,
                user_question=question,
                chat_history=chat_history,
                agent_name=agent_name,
                num_chat_history_in_prompt=num_chat_history_in_prompt,
                action_item_meet=action_item_meet,
                action_item_website=action_item_website,
            )
        )

        required_text = {
            "product_description": presentation_instance.to_json(to_str=True),
            "additional_information": additional_information,
            "covered_slides": covered_slides,
            "agent_name": agent_name,
            "action_item_rules": action_items,
        }

        response = self.inprompt_chain(
            question=question,
            chat_history_str=chat_history_str,
            required_texts=required_text,
            stream=stream,
            prompt=prompt_visual_sales_agent_v36,
            response_format=JSON_RESPONSE_FORMAT,
            tools=tools,
        )

        if response.get("response_type", "response") == "function_call":
            return self._handle_metadata_variables(response)

        if stream:
            word_by_word_gen = decode_visual_sales_agent_answer_stream(response["response"])
            slide_id = next(word_by_word_gen)
            response.update({"response": word_by_word_gen})

            response = self.handle_stream(
                response=response,
                chat_history=chat_history_short,
                question=question,
                user_id=user_id,
                document_ids=document_ids,
                stream=stream,
                stream_sentence=stream_sentence,
                end_sign=end_sign,
                trim_end_sign=trim_end_sign,
                ranking=ranking,
                standalone=standalone,
                agent_name=agent_name,
                embedding_model=embedding_model,
                agent_uuid=agent_uuid,
                main_document_id=main_document_id,
            )

        else:
            in_prompt_response = json.loads(response["response"])
            response.update({"response": in_prompt_response["2_text"]})
            slide_id = in_prompt_response["1_used_reference_id"]

            response = self.handle_non_stream(
                response=response,
                chat_history=chat_history_short,
                question=question,
                user_id=user_id,
                document_ids=document_ids,
                stream=stream,
                ranking=ranking,
                standalone=standalone,
                agent_name=agent_name,
                embedding_model=embedding_model,
                agent_uuid=agent_uuid,
                main_document_id=main_document_id,
            )

        final_response = self._handle_metadata_variables(response, slide_id, presentation_instance_full)

        if final_response["metadata"].get("document_id", "None") == agent_uuid:
            final_response["metadata"]["document_id"] = "current"

        return final_response
