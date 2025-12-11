# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import logging
import os
from typing import Optional
import firebase_admin
from firebase_admin import firestore, credentials, firestore_async
from pathlib import Path

from src.models.chat import CachedResponse
from src.models.party import Party
from src.utils import load_env

load_env()

logger = logging.getLogger(__name__)

credentials_path = (
    "wahl-chat-firebase-adminsdk.json"
    if os.getenv("ENV") == "prod"
    else "wahl-chat-dev-firebase-adminsdk.json"
)

# If the credentials file does not exist, use the application default credentials
if Path(credentials_path).exists():
    cred = credentials.Certificate(credentials_path)
    firebase_admin.initialize_app(cred)
else:
    firebase_admin.initialize_app()

db = firestore.client()

async_db = firestore_async.client()


async def aget_parties() -> list[Party]:
    parties = async_db.collection("parties").stream()
    return [Party(**party.to_dict()) async for party in parties]


async def aget_party_by_id(party_id: str) -> Optional[Party]:
    party_ref = async_db.collection("parties").document(party_id)
    party = await party_ref.get()
    if party.exists:
        return Party(**party.to_dict())
    return None


async def aget_proposed_questions_for_party(party_id: str) -> list[str]:
    questions = async_db.collection(f"proposed_questions/{party_id}/questions").stream()
    return [question.get("content") async for question in questions]


async def aget_cached_answers_for_party(
    party_id: str, cache_key: str
) -> list[CachedResponse]:
    cached_answers = async_db.collection(
        f"cached_answers/{party_id}/{cache_key}"
    ).stream()
    return [
        CachedResponse(**cached_answer.to_dict())
        async for cached_answer in cached_answers
    ]


async def awrite_cached_answer_for_party(
    party_id: str, cache_key: str, cached_answer: CachedResponse
) -> None:
    cached_answer_ref = async_db.collection(
        f"cached_answers/{party_id}/{cache_key}"
    ).document()
    await cached_answer_ref.set(cached_answer.model_dump())


async def awrite_llm_status(is_at_rate_limit: bool) -> None:
    llm_status_ref = async_db.collection("system_status").document("llm_status")
    await llm_status_ref.set({"is_at_rate_limit": is_at_rate_limit})


async def aupdate_voice_transcription(
    session_id: str,
    grouped_message_id: str,
    message_id: str,
    transcribed_text: str,
) -> None:
    """
    Update Firebase with the transcribed text for a voice message.
    Path: chat_sessions/{session_id}/messages/{message_id}
    """
    doc_ref = (
        async_db.collection("chat_sessions")
        .document(session_id)
        .collection("messages")
        .document(grouped_message_id)
    )
    doc = await doc_ref.get()

    if not doc.exists:
        raise ValueError(f"Message {message_id} not found in session {session_id}")

    data = doc.to_dict()

    # Update the inner message content
    for msg in data["messages"]:
        if msg["id"] == message_id:
            msg["content"] = transcribed_text

    # Set transcription status to complete
    data["voice_transcription"] = {"status": "transcribed"}

    await doc_ref.update(data)


async def aupdate_voice_transcription_error(
    session_id: str,
    message_id: str,
    error_message: str,
) -> None:
    """
    Update Firebase with an error status for a failed voice transcription.
    Path: chat_sessions/{session_id}/messages/{message_id}
    """
    doc_ref = (
        async_db.collection("chat_sessions")
        .document(session_id)
        .collection("messages")
        .document(message_id)
    )
    doc = await doc_ref.get()

    if not doc.exists:
        return  # Don't raise on error path

    data = doc.to_dict()
    data["voice_transcription"] = {"status": "error", "error": error_message}

    await doc_ref.update(data)
