# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from pathlib import Path
from typing import Union
import logging

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from src.models.party import Party

from src.utils import load_env, safe_load_api_key

from src.chatbot_async import rerank_documents

load_env()

logger = logging.getLogger(__name__)

BASE_PATH = Path(__file__).parent
EMBEDDING_SIZE = 3072  # Embedding sizes for the OpenAI models: https://platform.openai.com/docs/guides/embeddings#how-to-get-embeddings

# Get environment suffix
env = os.getenv("ENV", "dev")
env_suffix = f"_{env}" if env in ["prod", "dev"] else "_dev"

PARTY_INDEX_NAME = f"all_parties{env_suffix}"
VOTING_BEHAVIOR_INDEX_NAME = f"justified_voting_behavior{env_suffix}"
PARLIAMENTARY_QUESTIONS_INDEX_NAME = f"parliamentary_questions{env_suffix}"

embed = OpenAIEmbeddings(
    model="text-embedding-3-large", openai_api_key=safe_load_api_key("OPENAI_API_KEY")
)

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Initialize Qdrant vector stores
qdrant_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=PARTY_INDEX_NAME,
    embedding=embed,
    vector_name="dense",
    content_payload_key="text",
)
voting_behavior_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=VOTING_BEHAVIOR_INDEX_NAME,
    embedding=embed,
    vector_name="dense",
    content_payload_key="text",
)
parliamentary_questions_vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=PARLIAMENTARY_QUESTIONS_INDEX_NAME,
    embedding=embed,
    vector_name="dense",
    content_payload_key="text",
)


async def _identify_relevant_documents(
    vector_store: QdrantVectorStore,
    namespace: str,
    rag_query: str,
    n_docs: int = 5,
    score_threshold: float = 0.5,
) -> list[Document]:
    """
    Identify relevant documents based on the provided query and namespace.
    Uses direct Qdrant client to ensure all metadata is preserved.
    """
    # Get query vector
    query_vector = await embed.aembed_query(rag_query)

    # Create filter for the namespace
    filter_condition = Filter(
        must=[FieldCondition(key="namespace", match=MatchValue(value=namespace))]
    )

    # Search directly using Qdrant client to preserve all metadata
    # Note: Using sync client in async context - this might need optimization later
    search_result = qdrant_client.search(
        collection_name=vector_store.collection_name,
        query_vector=("dense", query_vector),
        limit=n_docs,
        with_payload=True,
        query_filter=filter_condition,
        score_threshold=score_threshold,
    )

    # Create LangChain Documents manually to preserve all metadata
    documents = []
    for point in search_result:
        if point.payload is None:
            continue

        # Extract content from text field
        content = point.payload.get("text", "")

        # Extract metadata (everything except text)
        metadata = {k: v for k, v in point.payload.items() if k != "text"}

        # Create Document with proper content and metadata
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents


async def identify_relevant_docs(
    party: Party,
    rag_query: str,
    n_docs: int = 5,
    score_threshold: float = 0.5,
) -> list[Document]:
    return await _identify_relevant_documents(
        vector_store=qdrant_vector_store,
        namespace=party.party_id,
        rag_query=rag_query,
        n_docs=n_docs,
        score_threshold=score_threshold,
    )


# relevant docs with reranking
async def identify_relevant_docs_with_reranking(
    party: Party,
    rag_query: str,
    n_docs: int = 20,
    score_threshold: float = 0.5,
) -> list[Document]:
    relevant_docs = await _identify_relevant_documents(
        vector_store=qdrant_vector_store,
        namespace=party.party_id,
        rag_query=rag_query,
        n_docs=n_docs,
        score_threshold=score_threshold,
    )

    # For now, return without external reranking since we're moving away from Pinecone
    # TODO: Implement alternative reranking if needed
    return relevant_docs[:5]  # Return top 5 documents


async def identify_relevant_docs_with_llm_based_reranking(
    party: Party,
    rag_query: str,
    chat_history: str,
    user_message: str,
    n_docs: int = 20,
    score_threshold: float = 0.5,
) -> list[Document]:
    relevant_docs = await _identify_relevant_documents(
        vector_store=qdrant_vector_store,
        namespace=party.party_id,
        rag_query=rag_query,
        n_docs=n_docs,
        score_threshold=score_threshold,
    )

    # Note: We lose the score information when using direct Qdrant search
    # If score sorting is critical, we could modify _identify_relevant_documents
    # to return scores as well

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
