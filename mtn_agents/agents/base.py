from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel
from functools import lru_cache

from ..constants import CONVERSATION_START_QUESTION, ActionItems
from ..chains import MasterChain, BaseChain
from ..types import ChainResponse, AgentResponseDict
from ..llm import BaseLLM, query_stream_sentence
from .utils import stream_sentence_with_switch
from ..utils import Presentation, ChatHistory
from ..retriever import BaseIndex


class BaseAgent(BaseModel):
    inprompt_chain: Optional[BaseChain] = None
    master_chain: Optional[BaseChain] = None
    use_master_chain: Optional[bool] = True

    class Config:
        arbitrary_types_allowed = True

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
        self.inprompt_chain = BaseChain(model=model, generator=generator, retriever=retriever)
        self.master_chain = MasterChain(generator=generator, retriever=retriever)
        self.use_master_chain = use_master_chain

    def __call__(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def handle_non_stream(
        self,
        response: ChainResponse,
        chat_history: Optional[str],
        question: Optional[str],
        user_id: Optional[str],
        document_ids: Optional[List[str]],
        stream: Optional[bool],
        ranking: Optional[bool],
        standalone: Optional[bool],
        agent_name: Optional[str],
        embedding_model: Optional[str],
        agent_uuid: Optional[str],
        main_document_id: Optional[str],
    ) -> ChainResponse:
        if not self.use_master_chain:
            return response
        else:
            if "I don't know.$eos$" in response["response"]:
                master_chain_response = self.master_chain(
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

                if master_chain_response["response_type"] == "function_call":
                    return master_chain_response

                response.update(master_chain_response)

            return response

    def handle_stream(
        self,
        response: ChainResponse,
        chat_history: Optional[str],
        question: Optional[str],
        user_id: Optional[str],
        document_ids: Optional[List[str]],
        stream: Optional[bool],
        stream_sentence: Optional[bool],
        end_sign: Optional[str],
        trim_end_sign: Optional[bool],
        ranking: Optional[bool],
        standalone: Optional[bool],
        agent_name: Optional[str],
        embedding_model: Optional[str],
        agent_uuid: Optional[str],
        main_document_id: Optional[str],
    ) -> ChainResponse:
        word_by_word_stream = response["response"]

        if not self.use_master_chain:
            if stream_sentence:
                sent_by_sent_stream = query_stream_sentence(
                    text=word_by_word_stream, end_sign=end_sign, trim=trim_end_sign
                )
                response.update({"response": sent_by_sent_stream})
                return response
            else:
                return response
        else:
            sent_by_sent_stream = stream_sentence_with_switch(
                word_by_word_generator=word_by_word_stream,
                end_sentence_sign=end_sign,
                clean_each_sentence=trim_end_sign,
                idk_list=["I don't know.$eos$"],
            )

            inprompt_knows_answer_switch = next(sent_by_sent_stream)
            if not inprompt_knows_answer_switch:
                master_chain_stream = self.master_chain(
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
                if master_chain_stream["response_type"] == "function_call":
                    return master_chain_stream

                if stream_sentence:
                    master_chain_stream_sent = query_stream_sentence(
                        text=master_chain_stream["response"], end_sign=end_sign, trim=trim_end_sign
                    )
                    master_chain_stream.update({"response": master_chain_stream_sent})

                response.update(master_chain_stream)

            else:
                if stream_sentence:
                    response.update({"response": sent_by_sent_stream})

            return response


class VisualBaseAgent(BaseAgent):
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

    @staticmethod
    @lru_cache(maxsize=None)
    def get_action_items(action_item_website: bool, action_item_meet: bool) -> str:
        items = []
        if action_item_website:
            items.append(ActionItems.WEBSITE_RULE)
        if action_item_meet:
            items.append(ActionItems.MEET_RULE)
        return ActionItems.HEADER + "\n" + "\n".join(items) if items else ""

    def prepare_data(
        self,
        input_product_description: Presentation,
        user_question: str,
        chat_history: ChatHistory,
        agent_name: str,
        num_chat_history_in_prompt: int,
        action_item_website: bool,
        action_item_meet: bool,
        first_message: str = "",
    ) -> Tuple[Presentation, str, str, List[str], str]:
        if first_message:
            user_first_message = first_message.format(agent_name=agent_name)
        else:
            user_first_message = CONVERSATION_START_QUESTION.format(agent_name=agent_name)

        if not chat_history:
            return self._handle_empty_chat_history(
                input_product_description, user_first_message, action_item_website, action_item_meet
            )

        if not chat_history[0].user or chat_history[0].user == "{}":
            chat_history.add_first_user_message(user_first_message=user_first_message)

        action_items = self.get_action_items(action_item_website, action_item_meet)

        filtered_chat_history = []
        for i, item in enumerate(chat_history):
            if i == len(chat_history) - 1:
                if (
                    item.assistant_metadata.get("chain") == "inprompt_chain"
                    and item.assistant_metadata.get("document_id") == "current"
                ):
                    item.slide_id = item.assistant_metadata.get("slide_id")
                else:
                    item.slide_id = "None"
                    item.assistant_metadata["slide_id"] = "None"
                    item.assistant = "I don't know.$eos$"
                filtered_chat_history.append(item)
            elif item.assistant_metadata and item.assistant_metadata.get("chain") == "inprompt_chain":
                filtered_chat_history.append(item)
        new_chat_history = ChatHistory(filtered_chat_history)

        covered_slides = self._get_covered_slides(new_chat_history)
        new_product_description = self.get_updated_product_description(input_product_description, new_chat_history)

        new_chat_history = new_chat_history[-num_chat_history_in_prompt:]
        chat_history_str = new_chat_history.to_str(assistant_name=agent_name, visual=True, chat_history=new_chat_history)

        # print(f"""\n\n{chat_history_str}\n\n""")

        return (new_product_description, user_question, chat_history_str, covered_slides, action_items, new_chat_history)

    def _handle_empty_chat_history(
        self,
        input_product_description: Optional[Presentation],
        user_first_message: str,
        action_item_website: bool,
        action_item_meet: bool,
    ) -> Tuple[Presentation, str, str, List[str], str]:
        new_product_description = Presentation([])
        if input_product_description:
            new_product_description = Presentation(
                [input_product_description["page_1"], input_product_description.get_next_slide("page_1")]
            )
        return (
            new_product_description,
            user_first_message,
            "",
            [],
            self.get_action_items(action_item_website, action_item_meet),
            ChatHistory([]),
        )

    @staticmethod
    def _get_covered_slides(chat_history: ChatHistory) -> List[str]:
        seen = set()
        covered_slides = []
        for item in chat_history:
            slide_id = item.slide_id
            if slide_id not in seen and slide_id not in ["", "None", None]:
                seen.add(slide_id)
                covered_slides.append(slide_id)
        if covered_slides:
            covered_slides.sort(key=lambda x: int(x.split('_')[1]))
        return covered_slides

    @staticmethod
    def get_updated_product_description(
        input_product_description: Optional[Presentation], chat_history: ChatHistory
    ) -> Presentation:
        if not input_product_description:
            return Presentation([])
        if chat_history:
            if chat_history[-1].slide_id != "None":
                latest_slide = chat_history[-1].slide_id
            else:
                for item in chat_history[::-1]:
                    if item.slide_id != "None":
                        latest_slide = item.slide_id
                        break
                else:
                    latest_slide = "page_1"
        else:
            latest_slide = "page_1"
        prev_slide = input_product_description.get_previous_slide(latest_slide)
        current_slide = input_product_description[latest_slide]
        next_slide = input_product_description.get_next_slide(latest_slide)
        next_next_slide = input_product_description.get_next_slide(next_slide.slide_number) if next_slide else ""

        return Presentation([item for item in [prev_slide, current_slide, next_slide, next_next_slide] if item])

    @staticmethod
    def _handle_metadata_variables(
        response: Dict[str, Any], slide_id_str: Optional[str] = "None", product_description: Optional[Presentation] = None
    ) -> AgentResponseDict:
        metadata = response.setdefault("metadata", {})
        metadata.setdefault("chain", "inprompt_chain")
        slide_id = metadata.get("slide_id", slide_id_str)

        slide_id_parts = slide_id.split("##")

        # MasterChain
        if len(slide_id_parts) > 1:
            document_id, *slide_id_part = slide_id_parts
            # Not presentable content
            if document_id == "document_1":
                metadata.update({"slide_id": slide_id_part[0], "document_id": "None", "url": "None"})
            # Url content
            elif response["metadata"].get("url", "None") != "None":
                metadata.update({"slide_id": slide_id_part[0], "document_id": document_id, "url": response["metadata"]["url"]})
            # Presentable content
            else:
                metadata.update({"slide_id": slide_id_part[0], "document_id": document_id, "url": "None"})

        # InPromptChain
        else:
            if product_description:
                temp_url = dict(product_description.get_slide(slide_id)).get(
                    "url", response["metadata"].get("url", "None")
                )
            else:
                temp_url = "None"
            document_id = "current"
            # Url content
            metadata.update({"slide_id": slide_id, "document_id": document_id, "url": temp_url})

        return response
