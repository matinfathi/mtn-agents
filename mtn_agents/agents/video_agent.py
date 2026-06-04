from typing import Optional, List

from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from ..constants import LATEST_EMBEDDING_MODEL
from ..prompt_templates import prompt_video_agent
from ..types import AgentResponseDict
from ..utils import ChatHistory
from ..retriever import BaseIndex
from ..llm import BaseLLM
from .base import BaseAgent


class VideoAgent(BaseAgent):
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
        video_transcription: Optional[str] = "",
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
        Handles the conversation with the agent for video agent.

            Args:
                - question (Optional[str]): The user's question or input. Defaults to "".
                - chat_history (Optional[ChatHistory]): The chat history object. Defaults to None.
                - video_transcription (Optional[str]): The video transcript of the paused part. Defaults to "".
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
                - ranking (Optional[bool]): Indicates if the references in metadata should be in order of relevance.
                - standalone (Optional[bool]): Indicates if we create standalone question for references.

            Returns:
                AgentResponse: The agent's response.
        """
        if chat_history:
            chat_history = chat_history[-num_chat_history_in_prompt:]
        else:
            chat_history = ChatHistory([])

        required_text = {
            "product_description": product_description,
            "additional_information": additional_information,
            "video_transcription": video_transcription,
            "agent_name": agent_name,
        }

        inprompt_switch = True if product_description else False

        if inprompt_switch:
            response = self.inprompt_chain(
                question=question,
                chat_history_str=chat_history.to_str(assistant_name=agent_name),
                required_texts=required_text,
                stream=stream,
                prompt=prompt_video_agent,
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
        else:
            response = self.master_chain(
                question=question,
                chat_history=chat_history,
                user_id=user_id,
                document_ids=document_ids,
                ranking=ranking,
                standalone=standalone,
                stream=stream,
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
