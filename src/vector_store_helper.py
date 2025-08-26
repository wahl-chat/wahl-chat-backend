# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from pathlib import Path
from typing import Union
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone
from src.models.party import Party

from src.utils import load_env, safe_load_api_key

from src.chatbot_async import rerank_documents

load_env()

logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent
EMBEDDING_SIZE = 3072  # Embedding sizes for the OpenAI models: https://platform.openai.com/docs/guides/embeddings#how-to-get-embeddings
PARTY_INDEX_NAME = "all-parties-index"
VOTING_BEHAVIOR_INDEX_NAME = "justified-voting-behavior-index"
PARLIAMENTARY_QUESTIONS_INDEX_NAME = "parliamentary-questions-index"

embed = OpenAIEmbeddings(
    model="text-embedding-3-large", openai_api_key=safe_load_api_key("OPENAI_API_KEY")
)

pc = Pinecone(pinecone_api_key=os.getenv("PINECONE_API_KEY"), embedding=embed)

party_index = pc.Index(PARTY_INDEX_NAME)
voting_behavior_index = pc.Index(VOTING_BEHAVIOR_INDEX_NAME)
parliamentary_questions_index = pc.Index(PARLIAMENTARY_QUESTIONS_INDEX_NAME)

pinecone_vector_store = PineconeVectorStore(index=party_index, embedding=embed)
voting_behavior_vector_store = PineconeVectorStore(
    index=voting_behavior_index, embedding=embed
)
parliamentary_questions_vector_store = PineconeVectorStore(
    index=parliamentary_questions_index, embedding=embed
)


async def _identify_relevant_documents(
    vector_store: PineconeVectorStore,
    namespace: str,
    rag_query: str,
    n_docs: int = 5,
    score_threshold: float = 0.5,
) -> list[Document]:
    """
    Identify relevant documents based on the provided query and namespace.
    """
    relevant_docs_with_scores = (
        await vector_store.asimilarity_search_with_relevance_scores(
            rag_query,
            namespace=namespace,
            k=n_docs,
            score_threshold=score_threshold,
        )
    )
    relevant_docs = [doc for doc, _ in relevant_docs_with_scores]
    return relevant_docs


async def identify_relevant_docs(
    party: Party,
    rag_query: str,
    n_docs: int = 5,
    score_threshold: float = 0.5,
) -> list[Document]:
    relevant_docs_with_scores = (
        await pinecone_vector_store.asimilarity_search_with_relevance_scores(
            rag_query,
            namespace=party.party_id,
            k=n_docs,
            score_threshold=score_threshold,
        )
    )
    relevant_docs = [doc for doc, _ in relevant_docs_with_scores]
    return relevant_docs


# relevant docs with reranking
async def identify_relevant_docs_with_reranking(
    party: Party,
    rag_query: str,
    n_docs: int = 20,
    score_threshold: float = 0.5,
) -> list[Document]:
    relevant_docs_with_scores = (
        await pinecone_vector_store.asimilarity_search_with_relevance_scores(
            rag_query,
            namespace=party.party_id,
            k=n_docs,
            score_threshold=score_threshold,
        )
    )
    relevant_docs = [doc for doc, _ in relevant_docs_with_scores]
    # dict matching id to document
    if len(relevant_docs) != 0:
        relevant_docs_matching_dict = {}
        relevant_docs_pinecone_list = []
        for index, doc in enumerate(relevant_docs):
            relevant_docs_matching_dict[f"{index}"] = doc
            relevant_docs_pinecone_list.append(
                {"id": str(index), "text": doc.page_content}
            )

        results = pc.inference.rerank(
            model="cohere-rerank-3.5",
            query=rag_query,
            documents=relevant_docs_pinecone_list,
            top_n=5,
            return_documents=True,
        )

        final_docs = [
            relevant_docs_matching_dict[element["document"]["id"]]
            for element in results.data
        ]
        return final_docs
    else:
        return relevant_docs


async def identify_relevant_docs_with_llm_based_reranking(
    party: Party,
    rag_query: str,
    chat_history: str,
    user_message: str,
    n_docs: int = 20,
    score_threshold: float = 0.5,
) -> list[Document]:
    relevant_docs_with_scores = (
        await pinecone_vector_store.asimilarity_search_with_relevance_scores(
            rag_query,
            namespace=party.party_id,
            k=n_docs,
            score_threshold=score_threshold,
        )
    )
    # sort the relevant docs by score in descending order to be sure that the most relevant docs are up top
    relevant_docs_with_scores = sorted(
        relevant_docs_with_scores, key=lambda x: x[1], reverse=True
    )
    relevant_docs = [doc for doc, _ in relevant_docs_with_scores]
    # construct string to pass to llm
    if len(relevant_docs) >= 5:
        # get indices of relevant docs
        relevant_docs = await rerank_documents(
            relevant_docs=relevant_docs,
            user_message=user_message,
            chat_history=chat_history,
        )
        return relevant_docs
    else:
        return relevant_docs


async def identify_relevant_votes(
    rag_query: str, n_docs: int = 5, score_threshold: float = 0.5
) -> list[Document]:
    """
    Identify relevant votes based on the provided query.

    :param rag_query: The query to search for relevant documents.
    :param n_docs: The number of documents to return.
    :param score_threshold: The score threshold for the similarity search.
    :return: A list of relevant documents.
    """
    return await _identify_relevant_documents(
        vector_store=voting_behavior_vector_store,
        namespace="vote_summary",
        rag_query=rag_query,
        n_docs=n_docs,
        score_threshold=score_threshold,
    )


async def identify_relevant_parliamentary_questions(
    party: Union[Party, str],
    rag_query: str,
    n_docs: int = 5,
    score_threshold: float = 0.7,
) -> list[Document]:
    """
    Identify relevant parliamentary questions based on the provided query and party.
    """
    namespace = f"{party.party_id if isinstance(party, Party) else party}-parliamentary-questions"
    return await _identify_relevant_documents(
        vector_store=parliamentary_questions_vector_store,
        namespace=namespace,
        rag_query=rag_query,
        n_docs=n_docs,
        score_threshold=score_threshold,
    )
