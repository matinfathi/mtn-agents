from mtn_agents.llm import OpenAILLM
from pinecone.core.openapi.data.models import QueryResponse

import numpy as np

from typing import List, Optional, Any, Dict


def find_cosine_similarity(vector: np.ndarray, matrix: np.ndarray):
    """Calculate the cosine similarity of a vector and every row of a matrix."""
    vector_norm = vector / np.linalg.norm(vector, axis=1, keepdims=True)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    # Compute cosine similarity
    cosine_similarities = np.dot(matrix_norm, vector_norm.T).flatten()

    return cosine_similarities


def ranking(
    answer: str, ref_list: List[str], vec_list: Optional[List[List[float]]] = None
) -> List[str]:
    """Find the target chunk of reference list."""
    if vec_list is None:
        vec_list = OpenAILLM().embedding(ref_list)

    if len(ref_list) != len(vec_list):
        raise ValueError(
            "Length of the references is not equal to the length of the vectors."
        )

    answer_vector = OpenAILLM().embedding([answer])
    vector = np.array(answer_vector)
    matrix = np.array(vec_list)

    cosine_similarities = find_cosine_similarity(vector, matrix)

    sorted_indices = np.argsort(cosine_similarities)[::-1]  # Sort in descending order
    sorted_ref_list = [ref_list[i] for i in sorted_indices]

    return sorted_ref_list


def ranking_pinecone(text: str, ref_list: QueryResponse) -> List[Dict[Any, Any]]:
    texts, values = [], []
    for item in ref_list:
        texts.append(item["metadata"]["text"])
        values.append(item["values"])

    answer_vector = OpenAILLM().embedding([text])
    vector = np.array(answer_vector)
    matrix = np.array(values)

    cosine_similarities = find_cosine_similarity(vector, matrix)

    sorted_indices = np.argsort(cosine_similarities)[::-1]  # Sort in descending order
    sorted_ref_list = [ref_list[i] for i in sorted_indices]

    return sorted_ref_list
