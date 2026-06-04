from pinecone import Pinecone
from pinecone.core.openapi.data.models import QueryResponse

from mtn_agents.constants import LATEST_EMBEDDING_MODEL
from mtn_agents.llm import OpenAILLM
from mtn_agents.utils import hex_uuid
from .base import BaseIndex

from typing import List, Dict, Union, Any, Optional
import os

TextOrDict = Union[str, Dict[str, Union[str, List[str]]]]
ComplexList = List[Dict[str, TextOrDict]]


class PineconeIndex(BaseIndex):
    index: Pinecone.Index = None
    index_name: str = ""
    client: Pinecone = None

    def __init__(self, pinecone_api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if api_key is None:
            raise ValueError("Pinecone API key cannot be 'None'.")

        try:
            self.client = Pinecone(api_key=api_key)
        except Exception as e:
            raise ValueError(
                f"Pinecone API client failed to initialize. Error: {e}"
            ) from e

    def load_index(self, index_name: Optional[str] = None):
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        try:
            self.index = self.client.Index(self.index_name)
        except Exception as e:
            raise Exception(f"Pinecone index failed to load. Error: {e}") from e

    def detailed_upsert(
        self,
        vectors: List[Dict[str, Union[str, float]]],
        user_id: str,
        batch_size: int = 100,
    ) -> None:
        if self.index is None:
            self.load_index()
        try:
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                self.index.upsert(vectors=batch, namespace=user_id)
        except Exception as e:
            raise Exception(f"Pinecone upsert failed. Error {e}") from e

    def upsert(
        self,
        texts: List[str],
        user_id: str,
        document_id: str,
    ) -> None:
        vectors = []
        embeddings = OpenAILLM().embedding(list_texts=texts)

        for idx, text in enumerate(texts):
            vectors.append(
                {
                    "id": hex_uuid(),
                    "values": embeddings[idx],
                    "metadata": {
                        "document_id": document_id,
                        "text": text,
                    },
                }
            )

        self.detailed_upsert(vectors, user_id)

    def detailed_query(
        self,
        input_texts: List[str],
        user_id: str = "",
        filter_dict: Union[Dict[str, Any], None] = None,
        top_k: int = 3,
        embedding_model: str = LATEST_EMBEDDING_MODEL,
        **kwargs,
    ) -> QueryResponse:
        if self.index is None:
            self.load_index()

        text_embeddings = OpenAILLM().embedding(list_texts=input_texts, model=embedding_model)
        try:
            output = self.index.query(
                vector=text_embeddings,
                filter=filter_dict,
                top_k=top_k,
                namespace=user_id,
                include_metadata=True if filter_dict else False,
                **kwargs,
            )
            return output

        except Exception as e:
            raise Exception(f"Pinecone query failed. Error {e}") from e

    def query(
        self, text: str, user_id: str, document_ids: List[str], top_k: int = 3, embedding_model: str = LATEST_EMBEDDING_MODEL
    ) -> QueryResponse:
        """
        Executes a query against a set of documents, returning the top K most relevant responses.

        This function performs a pinecone search query using a specified text input, filtering the results
        based on a list of document identifiers to ensure the responses are contextually relevant. It
        leverages a more detailed querying method internally to handle the complexity of the search and
        ranking process.

        Args:
            - text (str): The text query input by the user.
            - user_id (str): The unique identifier of the user making the query.
            - document_ids (List[str]): A list of document IDs to constrain the query, ensuring that only
                documents with these IDs are considered in the response.
            - top_k (int, optional): The number of top responses to return. Defaults to 3.

        Returns:
            - QueryResponse: A pinecone object contains required data and information about results.
        """
        metadata = {"document_id": {"$in": document_ids}}
        output = self.detailed_query(
            input_texts=[text], user_id=user_id, filter_dict=metadata, top_k=top_k, embedding_model=embedding_model
        )

        return output

    def query_return_text(
        self,
        text: str,
        user_id: str,
        document_ids: List[str],
        top_k: int = 3,
        threshold: float = 0.8,
    ) -> str:
        """
        Executes a query against a set of documents, returning the top K most relevant responses.

        This function performs a pinecone search query using a specified text input, filtering the results
        based on a list of document identifiers to ensure the responses are contextually relevant. It
        leverages a more detailed querying method internally to handle the complexity of the search and
        ranking process.

        Args:
            - text (str): The text query input by the user.
            - user_id (str): The unique identifier of the user making the query.
            - document_ids (List[str]): A list of document IDs to constrain the query, ensuring that only
                documents with these IDs are considered in the response.
            - top_k (int, optional): The number of top responses to return. Defaults to 3.
            - threshold (float, optional): The threshold to filter non-similar texts.

        Returns:
            - str: A string contains references based on input text.
        """
        if self.index is None:
            self.load_index()

        response = self.query(text, user_id, document_ids, top_k)

        texts = "\n\n".join(
            [
                item["metadata"]["text"]
                for item in response["matches"]
                if item["score"] >= threshold
            ]
        )

        return texts.strip()

    def delete(self, ids_: List[str], user_id: str = ""):
        if self.index is None:
            self.load_index()

        try:
            self.index.delete(ids=ids_, namespace=user_id)
        except Exception as e:
            raise Exception(f"Pinecone deletion failed. Error {e}") from e

    def delete_user_id(self, user_id: str):
        if self.index is None:
            self.load_index()

        try:
            self.index.delete(delete_all=True, namespace=user_id)
        except Exception as e:
            raise Exception(f"Pinecone User ID deletion failed. Error {e}") from e

    def describe(self):
        if self.index is None:
            self.load_index()
        return self.client.describe_index(self.index_name)

    def freshness(self):
        if self.index is None:
            self.load_index()
        return self.index.describe_index_stats()

    @staticmethod
    def record(
        texts: List[str], metadata: List[Dict[str, Union[List[str], str]]], embedding_model: str
    ) -> List[Dict[str, Union[List[str], str]]]:
        if len(texts) != len(metadata):
            raise ValueError("The length of texts and info's are not equal.")
        if not texts:
            raise ValueError("The list of texts for converting to embedding is empty.")
        embeddings = OpenAILLM().embedding(texts, model=embedding_model)
        records = []
        for idx in range(len(embeddings)):
            metadata[idx]["text"] = texts[idx]
            records.append(
                {"id": hex_uuid()[:12], "values": embeddings[idx], "metadata": metadata[idx]}
            )

        return records

    def prepare_data_for_pinecone(
        self, data: List[Dict[str, Any]], embedding_model: Optional[str] = LATEST_EMBEDDING_MODEL
    ) -> List[Dict[str, Union[List[str], str]]]:
        texts = []
        metadata = []
        for record in data:
            texts.append(record.get("content", record.get("text", "")))
            record.pop("content", None)
            metadata.append(record)

        pinecone_data = self.record(texts, metadata, embedding_model)

        return pinecone_data

    @staticmethod
    def convert_vertex_restricts_to_pinecone(
        restricts: Dict[str, Union[str, List[str]]]
    ):
        alternates = {"labels": "document_id", "cache": "cache_id"}
        new_restricts = {}
        for k, v in restricts.items():
            new_restricts[alternates.get(k, k)] = {"$in": v}

    def __len__(self):
        if self.index is None:
            self.load_index()
        return self.index.describe_index_stats()["total_vector_count"]
