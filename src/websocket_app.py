# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import asyncio
from datetime import datetime
import logging
import os
import random
from typing import List, Dict, Optional, Union
import aiohttp
import json

import socketio
import openai
from langchain_core.messages import BaseMessageChunk
from langchain_core.documents import Document
from pydantic import ValidationError


from src.chatbot_async import (
    generate_chat_title_and_chick_replies,
    generate_swiper_assistant_response,
    get_improved_rag_query_voting_behavior,
    get_question_targets_and_type,
    generate_pro_con_perspective,
    generate_improvement_rag_query,
    generate_streaming_chatbot_response,
    generate_chat_summary,
    generate_streaming_chatbot_comparing_response,
    generate_party_vote_behavior_summary,
)
from src.firebase_service import (
    aget_cached_answers_for_party,
    aget_parties,
    aget_party_by_id,
    aget_proposed_questions_for_party,
    awrite_cached_answer_for_party,
)
from src.models.chat import CachedResponse, GroupChatSession, Message, Role
from src.models.dtos import (
    ChatResponseCompleteDto,
    ChatSessionInitializedDto,
    PartyResponseChunkDto,
    PartyResponseCompleteDto,
    ChatUserMessageDto,
    InitChatSessionDto,
    ProConPerspectiveRequestDto,
    ProConPerspectiveDto,
    QuickRepliesAndTitleDto,
    RespondingPartiesDto,
    SourcesDto,
    Status,
    StatusIndicator,
    RequestSummaryDto,
    SummaryDto,
    VotingBehaviorRequestDto,
    VotingBehaviorDto,
    Vote,
    VotingBehaviorSummaryChunkDto,
    VotingBehaviorVoteDto,
    WahlChatSwiperResponseCompleteDto,
    WahlChatSwiperUserMessageDto,
)

from src.models.party import WAHL_CHAT_PARTY, Party
from src.vector_store_helper import (
    identify_relevant_votes,
    identify_relevant_docs_with_llm_based_reranking,
)
from src.utils import (
    build_chat_history_string,
    get_chat_history_hash_key,
    get_cors_allowed_origins,
    sanitize_references,
)

MAX_RESPONSE_CHUNK_LENGTH = 10

logger = logging.getLogger(__name__)
socketio_logger = logging.getLogger("socketio.asyncserver")
sio = socketio.AsyncServer(
    logger=socketio_logger,
    async_handlers=True,
    async_mode="aiohttp",
    allow_upgrades=True,
    monitor_clients=True,
    cors_allowed_origins=get_cors_allowed_origins(os.getenv("ENV")),
    always_connect=False,
    transports=["websocket"],
)


@sio.event
async def connect(sid: str, environ: dict):
    logger.info(f"Client connected: {sid}")


@sio.event
async def disconnect(sid: str, reason: str):
    if reason == sio.reason.CLIENT_DISCONNECT:
        logger.info(f"The client disconnected: {sid}")
    elif reason == sio.reason.SERVER_DISCONNECT:
        logger.info(f"The server disconnected the client: {sid}")
    else:
        logger.info(f"Disconnect reason for client {sid}: {reason}")
    # Remove the chat session from the session data
    async with sio.session(sid) as session:
        if "chat_sessions" in session:
            logger.info(f"Removing chat session data for client {sid}")
            del session["chat_sessions"]
    logger.info(f"Client disconnected: {sid}")


@sio.on("home")
async def home(sid: str, body: dict):
    await sio.emit("home_response", {"message": "Welcome to the wahl.chat API"}, to=sid)


@sio.on("chat_session_init")
async def init_chat_session(sid: str, body: dict):
    logger.info(f"Client {sid} requested chat session initialization with body: {body}")
    try:
        create_session_dto = InitChatSessionDto(**body)
    except ValidationError as e:
        logger.error(
            f"Error validating chat session initialization for client {sid}: {e}"
        )
        chat_session_initialized_dto = ChatSessionInitializedDto(
            session_id=None,
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit(
            "chat_session_initialized",
            chat_session_initialized_dto.model_dump(),
            to=sid,
        )
        return

    logger.debug(f"Creating group chat session: {create_session_dto}")

    chat_session = GroupChatSession(
        session_id=create_session_dto.session_id,
        chat_history=create_session_dto.chat_history,
        title=create_session_dto.current_title,
        chat_response_llm_size=create_session_dto.chat_response_llm_size,
        last_quick_replies=create_session_dto.last_quick_replies,
        is_cacheable=create_session_dto.is_cacheable,
    )

    async with sio.session(sid) as session:
        session["chat_sessions"] = session.get("chat_sessions", {})
        session["chat_sessions"][create_session_dto.session_id] = chat_session

    logger.debug(f"Chat session initialized for client {sid}")

    chat_session_initialized_dto = ChatSessionInitializedDto(
        session_id=create_session_dto.session_id,
        status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
    )
    await sio.emit(
        "chat_session_initialized",
        chat_session_initialized_dto.model_dump(),
        to=sid,
    )


# event to emit a chat summary based on fct from chatbot_async.py
@sio.on("chat_summary_request")
async def chat_summary_request(sid: str, body: dict):
    logger.info(f"Client {sid} requested chat summary from session_id: {body}")
    try:
        request_summary = RequestSummaryDto(**body)
        chat_history = request_summary.chat_history
    except ValidationError as e:
        logger.error(f"Error validating chat summary request for client {sid}: {e}")
        response_dto = SummaryDto(
            chat_summary="",
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit("chat_summary_complete", response_dto.model_dump(), to=sid)
        return

    try:
        chat_summary = await generate_chat_summary(chat_history)
        logger.debug(f"Chat summary generated: {chat_summary}")
        response_dto = SummaryDto(
            chat_summary=chat_summary,
            status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
        )
        await sio.emit("chat_summary_complete", response_dto.model_dump(), to=sid)
    except Exception as e:
        logger.error(
            f"Error generating chat summary for session {request_summary}: {e}"
        )
        response_dto = SummaryDto(
            chat_summary="Hier sollte eigentlich eine Zusammenfassung stehen...",
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit("chat_summary_complete", response_dto.model_dump(), to=sid)
        return


@sio.on("pro_con_perspective_request")
async def get_pro_con_perspective(sid: str, body: dict):
    logger.info(f"Client {sid} requested pro/con perspective with body: {body}")
    try:
        pro_con_assessment = ProConPerspectiveRequestDto(**body)
        party_id = pro_con_assessment.party_id
        last_user_message_str = pro_con_assessment.last_user_message
        last_assistant_message_str = pro_con_assessment.last_assistant_message
    except ValidationError as e:
        logger.error(
            f"Error validating pro/con perspective request for client {sid}: {e}"
        )
        response_dto = ProConPerspectiveDto(
            request_id=None,
            message=Message(role="assistant", content=""),
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit(
            "pro_con_perspective_complete", response_dto.model_dump(), to=sid
        )
        return

    logger.debug(
        f"Generating pro/con perspective for party {party_id} with user message '{last_user_message_str}' and assistant message '{last_assistant_message_str}'"
    )

    try:
        party = await aget_party_by_id(party_id)

        if party is None:
            raise ValueError(f"Party {party_id} not found")

        last_user_message = Message(role="user", content=last_user_message_str)
        last_assistant_message = Message(
            role="assistant", content=last_assistant_message_str
        )

        chat_history = [last_user_message, last_assistant_message]

        pro_con_perspective = await generate_pro_con_perspective(chat_history, party)

        logger.debug(f"Emitting pro/con perspective to client {sid}")

        response_dto = ProConPerspectiveDto(
            request_id=pro_con_assessment.request_id,
            message=pro_con_perspective,
            status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
        )

        await sio.emit(
            "pro_con_perspective_complete", response_dto.model_dump(), to=sid
        )
    except Exception as e:
        logger.error(
            f"Error generating pro/con perspective for party {party_id}: {e}",
            exc_info=True,
        )
        response_dto = ProConPerspectiveDto(
            request_id=pro_con_assessment.request_id,
            message=None,
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit(
            "pro_con_perspective_complete", response_dto.model_dump(), to=sid
        )
        return


async def emit_cached_party_response(
    sid: str,
    party: Party,
    group_chat_session: GroupChatSession,
    cached_response: CachedResponse,
):
    # Sleep for a short time to simulate processing time
    await asyncio.sleep(1)
    sources_dto = SourcesDto(
        session_id=group_chat_session.session_id,
        sources=cached_response.sources,
        party_id=party.party_id,
        rag_query=cached_response.rag_query,
    )
    await sio.emit("sources_ready", sources_dto.model_dump(), to=sid)

    full_response = cached_response.content
    # artificially chunk the response and emit it
    chunk_index = 0
    for i in range(0, len(full_response), MAX_RESPONSE_CHUNK_LENGTH):
        chunk = full_response[i : i + MAX_RESPONSE_CHUNK_LENGTH]
        chat_response_dto = PartyResponseChunkDto(
            session_id=group_chat_session.session_id,
            party_id=party.party_id,
            chunk_index=chunk_index,
            chunk_content=chunk,
            is_end=False,
        )
        await sio.emit(
            "party_response_chunk_ready", chat_response_dto.model_dump(), to=sid
        )
        chunk_index += 1
        await asyncio.sleep(0.025)
    # Emit a finalizing chunk
    chat_response_dto = PartyResponseChunkDto(
        session_id=group_chat_session.session_id,
        party_id=party.party_id,
        chunk_index=chunk_index,
        chunk_content="",
        is_end=True,
    )

    chatbot_message = Message(
        role="assistant",
        content=full_response,
        sources=cached_response.sources,
        party_id=party.party_id,
        current_chat_title=group_chat_session.title,
        quick_replies=[],
        rag_query=cached_response.rag_query,
    )
    group_chat_session.chat_history.append(chatbot_message)

    # Emit party response complete event
    party_response_complete_dto = PartyResponseCompleteDto(
        session_id=group_chat_session.session_id,
        party_id=party.party_id,
        complete_message=full_response,
        status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
    )
    logger.debug(f"Party response complete: {party_response_complete_dto}")
    await sio.emit(
        "party_response_complete", party_response_complete_dto.model_dump(), to=sid
    )
    logger.info(
        f"Party response {party_response_complete_dto.model_dump()} for {party.party_id} emitted to client {sid}"
    )


async def fetch_and_emit_party_response(
    sid: str,
    party: Party,
    conversation_history_str: str,
    question_for_party: str,
    group_chat_session: GroupChatSession,
    all_available_parties: List[Party],
    use_premium_llms: bool,
    is_proposed_question: bool = False,
    is_cacheable_chat: bool = True,
    # Change to None or dictionary for comparing scenario, or list for normal scenario
    # so we unify them in code with separate variables
    relevant_docs: Optional[Union[List[Document], Dict[str, List[Document]]]] = None,
    parties_being_compared: Optional[List[Party]] = None,
    is_comparing_question: bool = False,
    # for comparing scenario we need to pass List because all queries were computed in advance
    improved_rag_query_list: List[str] = [],
):
    # We’ll store single-party docs and multi-party docs separately:
    relevant_docs_list: Optional[List[Document]] = None
    relevant_docs_dict: Optional[Dict[str, List[Document]]] = None

    # For caching
    cache_key: Optional[str] = None
    cached_answer_to_emit: Optional[CachedResponse] = None
    cache_conversation_history_str = build_chat_history_string(
        group_chat_session.chat_history, all_available_parties
    )
    # full_response will be assigned a BaseMessageChunk later
    full_response: Optional[BaseMessageChunk] = None

    try:
        # Handle proposed question => possibility of picking a cached response
        logger.debug(
            f"Fetching party response for party {party.party_id}: is_proposed_question={is_proposed_question}, is_cacheable_chat={is_cacheable_chat}"
        )
        if is_proposed_question or is_cacheable_chat:
            if is_proposed_question:
                cache_key = question_for_party
            else:
                cache_key = get_chat_history_hash_key(cache_conversation_history_str)
            logger.debug(
                f"Checking cache for party {party.party_id} with cache key {cache_key}"
            )
            existing_cached_answers: List[
                CachedResponse
            ] = await aget_cached_answers_for_party(party.party_id, cache_key)
            logger.info(
                f"Fetched {len(existing_cached_answers)} cached answers for party {party.party_id} and cache_key {cache_key}"
            )

            cached_answer_limit = 1 if is_proposed_question else 1
            # Select a random cached answer to emit, if None is selected, the chatbot will generate a new response (and add it to the cache for proposed questions)
            # If there are at least `cached_answer_limit` cache answers, we'll always emit a cached answer
            possible_answers: list[CachedResponse | None] | list[CachedResponse] = (
                existing_cached_answers + [None]
                if len(existing_cached_answers) < cached_answer_limit
                else existing_cached_answers
            )
            cached_answer_to_emit = random.choice(possible_answers)
            logger.debug(
                f"Selected cached answer: {cached_answer_to_emit} (None means a new response will be generated)"
            )

        if cached_answer_to_emit is not None:
            logger.info(
                f"Selected cached answer for party {party.party_id} and question {question_for_party}: {cached_answer_to_emit}"
            )
            await emit_cached_party_response(
                sid,
                party,
                group_chat_session,
                cached_answer_to_emit,
            )
            return

        # If not is_comparing_question, we do a single-party RAG
        if not is_comparing_question:
            improved_rag_query = await generate_improvement_rag_query(
                party, conversation_history_str, question_for_party
            )
            logger.debug(f"Improved RAG query: {improved_rag_query}")

            # Identify relevant docs as a list
            relevant_docs_list = await identify_relevant_docs_with_llm_based_reranking(
                party=party,
                rag_query=improved_rag_query,
                chat_history=conversation_history_str,
                user_message=question_for_party,
            )
            # comparing scenario requires improved_rag_query to be a list, so match for both scenarios
            improved_rag_query_list = [improved_rag_query]

            logger.debug(f"Identified relevant docs (list): {relevant_docs_list}")

            sources = []
            for source_doc in relevant_docs_list:
                # Safely parse page_number. If it’s None, fallback to 0
                page_raw = source_doc.metadata.get("page", 0)
                page_number = int(page_raw if page_raw is not None else 0)
                # Shift by +1 for display indexing
                page_number += 1

                source = {
                    "source": source_doc.metadata.get("document_name"),
                    "page": page_number,
                    "document_publish_date": source_doc.metadata.get(
                        "document_publish_date"
                    ),
                    "url": source_doc.metadata.get("url"),
                    "source_document": source_doc.metadata.get("source_document"),
                }
                sources.append(source)

            sources_dto = SourcesDto(
                session_id=group_chat_session.session_id,
                party_id=party.party_id,
                rag_query=improved_rag_query_list,
                sources=sources,
            )
            await sio.emit("sources_ready", sources_dto.model_dump(), to=sid)

        else:
            # For comparing scenario, we assume relevant_docs is dict
            # but if your code still calls `identify_relevant_docs` for multiple parties,
            # adapt accordingly
            if relevant_docs is None:
                # Fallback to empty dict if no docs are provided
                relevant_docs_dict = {}
            else:
                # We assume user passed in a dict
                relevant_docs_dict = dict(relevant_docs)  # type: ignore

            logger.debug(f"Identified relevant docs (dict): {relevant_docs_dict}")
            sources = []
            if parties_being_compared:
                for rel_party in parties_being_compared:
                    for source_doc in relevant_docs_dict.get(rel_party.party_id, []):
                        page_raw = source_doc.metadata.get("page", 0)
                        page_number = int(page_raw if page_raw is not None else 0)
                        page_number += 1

                        source = {
                            "source": source_doc.metadata.get("document_name"),
                            "page": page_number,
                            "document_publish_date": source_doc.metadata.get(
                                "document_publish_date"
                            ),
                            "url": source_doc.metadata.get("url"),
                            "source_document": source_doc.metadata.get(
                                "source_document"
                            ),
                            "party_id": rel_party.party_id,
                        }
                        sources.append(source)

            sources_dto = SourcesDto(
                session_id=group_chat_session.session_id,
                party_id=party.party_id,
                rag_query=improved_rag_query_list,
                sources=sources,
            )
            await sio.emit("sources_ready", sources_dto.model_dump(), to=sid)

        # Now generate the answer stream
        if not is_comparing_question:
            chunk_stream = await generate_streaming_chatbot_response(
                party,
                conversation_history_str,  # list of Messages
                question_for_party,
                relevant_docs_list or [],  # pass an empty list if None
                all_parties=all_available_parties,
                chat_response_llm_size=group_chat_session.chat_response_llm_size,
                use_premium_llms=use_premium_llms,
            )
        else:
            chunk_stream = await generate_streaming_chatbot_comparing_response(
                party,
                conversation_history_str,  # list of Messages
                question_for_party,
                relevant_docs_dict or {},  # pass empty dict if None
                parties_being_compared or [],
                chat_response_llm_size=group_chat_session.chat_response_llm_size,
                use_premium_llms=use_premium_llms,
            )

        chunk_index = 0
        async for message_chunk in chunk_stream:
            if full_response is None:
                full_response = message_chunk
            else:
                full_response += message_chunk

            for i in range(0, len(message_chunk.content), MAX_RESPONSE_CHUNK_LENGTH):
                if i > 0:
                    # Sleep for a short time to simulate processing time
                    await asyncio.sleep(0.025)
                chunk_content = message_chunk.content[i : i + MAX_RESPONSE_CHUNK_LENGTH]
                chat_response_dto = PartyResponseChunkDto(
                    session_id=group_chat_session.session_id,
                    party_id=party.party_id,
                    chunk_index=chunk_index,
                    chunk_content=chunk_content,
                    is_end=False,
                )
                await sio.emit(
                    "party_response_chunk_ready", chat_response_dto.model_dump(), to=sid
                )
                chunk_index += 1

        # Emit a finalizing chunk
        chat_response_dto = PartyResponseChunkDto(
            session_id=group_chat_session.session_id,
            party_id=party.party_id,
            chunk_index=chunk_index,
            chunk_content="",
            is_end=True,
        )
        logger.debug(
            f"Emitting final chat response chunk {chat_response_dto} with index {chunk_index} "
            f"for party {party.party_id} to client {sid}"
        )
        await sio.emit(
            "party_response_chunk_ready", chat_response_dto.model_dump(), to=sid
        )

        # Build the full message
        if full_response is None:
            full_content = ""
        else:
            # Convert content to string if it's a list
            full_content = (
                str(full_response.content)
                if isinstance(full_response.content, list)
                else full_response.content
            )

        full_content = sanitize_references(full_content)

        chatbot_message = Message(
            role="assistant",
            content=full_content,
            sources=sources,  # from the branch above
            party_id=party.party_id,
            current_chat_title=group_chat_session.title,
            quick_replies=[],
            rag_query=improved_rag_query_list,
        )
        group_chat_session.chat_history.append(chatbot_message)

        # Emit party response complete event
        party_response_complete_dto = PartyResponseCompleteDto(
            session_id=group_chat_session.session_id,
            party_id=party.party_id,
            complete_message=full_content,
            status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
        )
        logger.debug(f"Party response complete: {party_response_complete_dto}")
        await sio.emit(
            "party_response_complete", party_response_complete_dto.model_dump(), to=sid
        )
        logger.info(
            f"Party response {party_response_complete_dto.model_dump()} for {party.party_id} emitted to client {sid}"
        )

        # If it was a proposed question and we generated something new, cache it
        if cache_key is not None and cached_answer_to_emit is None:
            logger.info(
                f"Writing generated response to cache for party {party.party_id} and cache key {cache_key}"
            )
            cached_answer = CachedResponse(
                content=full_content,
                sources=sources,
                rag_query=improved_rag_query_list,
                created_at=datetime.now(),
                cached_conversation_history=cache_conversation_history_str,
                depth=len(group_chat_session.chat_history),
                user_message_depth=len(
                    [m for m in group_chat_session.chat_history if m.role == Role.USER]
                ),
            )
            await awrite_cached_answer_for_party(
                party.party_id, cache_key, cached_answer
            )
            logger.debug(f"Written cached answer: {cached_answer}")
    except openai.BadRequestError as e:
        logger.error(
            f"Error fetching and emitting party response for {party.party_id}: {e}",
            exc_info=True,
        )
        party_response_complete_dto = PartyResponseCompleteDto(
            session_id=group_chat_session.session_id,
            party_id=party.party_id,
            complete_message="Diese Frage kann ich leider nicht beantworten.",
            status=Status(
                indicator=StatusIndicator.ERROR,
                message=str(e),
            ),
        )
    except Exception as e:
        logger.error(
            f"Error fetching and emitting party response for {party.party_id}: {e}",
            exc_info=True,
        )
        party_response_complete_dto = PartyResponseCompleteDto(
            session_id=group_chat_session.session_id,
            party_id=party.party_id,
            complete_message="Es tut mir Leid, leider ist ein Fehler aufgetreten. Bitte versuche es später erneut.",
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit(
            "party_response_complete", party_response_complete_dto.model_dump(), to=sid
        )
        return


async def process_party(
    party: Party,
    chat_history_str: str,
    general_question: str,
    relevant_doc_dict: Dict[str, List[Document]],
    lock: asyncio.Lock,
    improved_rag_query_list: List[str],
):
    logger.debug(
        f"For Party {party.party_id} the relevant docs are being identified by coroutine"
    )

    improved_rag_query = await generate_improvement_rag_query(
        party, chat_history_str, general_question
    )

    relevant_docs = await identify_relevant_docs_with_llm_based_reranking(
        party=party,
        rag_query=improved_rag_query,
        chat_history=chat_history_str,
        user_message=general_question,
    )

    # Safely update the shared improved_rag_query list
    async with lock:
        improved_rag_query_list.append(improved_rag_query)

    # Safely update the shared dictionary
    async with lock:
        relevant_doc_dict[party.party_id] = relevant_docs


@sio.on("chat_answer_request")
async def chat_answer_request(sid: str, body: dict):
    logger.info(f"Client {sid} requested chat answer with body: {body}")
    try:
        chat_message_data = ChatUserMessageDto(**body)
    except ValidationError as e:
        logger.error(f"Error validating chat message data for client {sid}: {e}")
        chat_response_complete_dto = ChatResponseCompleteDto(
            session_id=None,
            status=Status(
                indicator=StatusIndicator.ERROR,
                message=str(e),
            ),
        )
        await sio.emit(
            "chat_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    logger.debug(f"Chat message data: {chat_message_data}")

    # Extract user message
    user_message = Message(
        role="user",
        content=chat_message_data.user_message,
    )

    # Access chat session from socket session
    try:
        async with sio.session(sid) as session:
            chat_session: GroupChatSession = session.get("chat_sessions", {}).get(
                chat_message_data.session_id
            )
            if chat_session is None:
                raise ValueError(
                    f"Chat session with ID {chat_message_data.session_id} not found"
                )

            # Update session with user message
            chat_history = chat_session.chat_history
            # Append the user message if it not identical to the last message
            if (
                len(chat_history) == 0
                or chat_history[-1].content != user_message.content
            ):
                chat_history.append(user_message)

            # check if the user message is in the last quick replies
            last_quick_replies = chat_session.last_quick_replies
            is_beginning_of_chat = len(chat_history) == 1
            logger.debug(f"Is beginning of chat: {is_beginning_of_chat}")
            if (
                not is_beginning_of_chat
                and user_message.content not in last_quick_replies
            ):
                # after the first user message only chat sessions that only use quick replies are cacheable
                chat_session.is_cacheable = False
    except Exception as e:
        logger.error(
            f"Error accessing chat session for client {sid}: {e}", exc_info=True
        )
        chat_response_complete_dto = ChatResponseCompleteDto(
            session_id=chat_message_data.session_id,
            status=Status(
                indicator=StatusIndicator.ERROR,
                message="It seems like the chat session has not been started",
            ),
        )
        await sio.emit(
            "chat_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    all_parties = await aget_parties()
    pre_selected_parties = [
        party for party in all_parties if party.party_id in chat_message_data.party_ids
    ]

    pre_selected_party_ids = [party.party_id for party in pre_selected_parties]
    logger.debug(f"Pre-selected party IDs: {pre_selected_party_ids}")

    chat_history_without_last_user_message = chat_history[:-1]
    chat_history_str = build_chat_history_string(
        chat_history_without_last_user_message, all_parties
    )

    try:
        (
            party_id_list,
            general_question,
            is_comparing_question,
        ) = await get_question_targets_and_type(
            user_message=user_message.content,
            previous_chat_history=chat_history_str,
            all_available_parties=all_parties + [WAHL_CHAT_PARTY],
            currently_selected_parties=pre_selected_parties,
        )
    except openai.BadRequestError as e:
        logger.error(
            f"Error identifying question targets and type: {e}",
            exc_info=True,
        )
        responding_parties_dto = RespondingPartiesDto(
            session_id=chat_message_data.session_id,
            party_ids=[WAHL_CHAT_PARTY.party_id],
        )
        logger.debug(
            f"Emitting responding parties {responding_parties_dto.party_ids} to client {sid}"
        )
        await sio.emit(
            "responding_parties_selected",
            responding_parties_dto.model_dump(),
            to=sid,
        )
        party_response_complete_dto = PartyResponseCompleteDto(
            session_id=chat_session.session_id,
            party_id=WAHL_CHAT_PARTY.party_id,
            complete_message="Diese Frage kann ich leider nicht beantworten.",
            status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
        )
        await sio.emit(
            "party_response_complete", party_response_complete_dto.model_dump(), to=sid
        )
        chat_response_complete_dto = ChatResponseCompleteDto(
            session_id=chat_message_data.session_id,
            status=Status(
                indicator=StatusIndicator.ERROR,
                message=f"Error identifying question targets and type: {e}",
            ),
        )
        await sio.emit(
            "chat_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    logger.info(
        f"Identified question targets and type: party_id_list={party_id_list}, general_question={general_question}, is_comparing_question={is_comparing_question}"
    )

    if not party_id_list:
        logger.debug(f"No party IDs selected, defaulting to wahl-chat for client {sid}")
        party_id_list = ["wahl-chat"]
    elif is_beginning_of_chat and len(party_id_list) > 7:
        # If we are in the beginning of the chat, we only allow up to 7 party IDs for automatic selection
        # If more, we default to wahl-chat which will ask the user to select parties
        logger.debug(
            f"Too many party IDs selected at the beginning of the chat, defaulting to wahl-chat for client {sid}"
        )
        party_id_list = ["wahl-chat"]

    parties_to_respond = [
        party
        for party in all_parties + [WAHL_CHAT_PARTY]
        if party.party_id in party_id_list
    ]
    if not is_comparing_question:
        responding_parties_dto = RespondingPartiesDto(
            session_id=chat_message_data.session_id,
            party_ids=party_id_list,
        )
    else:
        responding_parties_dto = RespondingPartiesDto(
            session_id=chat_message_data.session_id,
            party_ids=["wahl-chat"],
        )
    logger.debug(
        f"Emitting responding parties {responding_parties_dto.party_ids} to client {sid}"
    )
    await sio.emit(
        "responding_parties_selected",
        responding_parties_dto.model_dump(),
        to=sid,
    )

    if len(parties_to_respond) == 1 or not is_comparing_question:
        party_coros = []
        for party in parties_to_respond:
            # get the proposed questions for the party
            proposed_questions_for_party = await aget_proposed_questions_for_party(
                party.party_id
            )
            proposed_questions_group = await aget_proposed_questions_for_party("group")

            is_proposed_question = (
                user_message.content in proposed_questions_for_party
                or user_message.content in proposed_questions_group
            )
            logger.debug(f"Is proposed question: {is_proposed_question}")
            if is_beginning_of_chat and not is_proposed_question:
                # chat sessions with custom initial questions are not cacheable
                chat_session.is_cacheable = False
            party_coros.append(
                fetch_and_emit_party_response(
                    sid,
                    party,
                    chat_history_str,
                    general_question,
                    chat_session,
                    all_available_parties=all_parties,
                    use_premium_llms=chat_message_data.user_is_logged_in,
                    is_proposed_question=is_proposed_question,
                    is_cacheable_chat=chat_session.is_cacheable,
                )
            )
    else:
        # chat sessions with comparison answers are not cacheable for now
        chat_session.is_cacheable = False

        parties_being_compared = parties_to_respond
        relevant_doc_dict: dict[str, list] = {}
        improved_rag_query_list: list[str] = []
        lock = asyncio.Lock()

        party_tasks = [
            process_party(
                party,
                chat_history_str,
                general_question,
                relevant_doc_dict,
                lock,
                improved_rag_query_list,
            )
            for party in parties_being_compared
        ]
        try:
            await asyncio.wait_for(
                asyncio.gather(*party_tasks),
                timeout=40,
            )
        except asyncio.TimeoutError as e:
            logger.error(f"Timeout while fetching the correct party documents: {e}")
            chat_response_complete_dto = ChatResponseCompleteDto(
                session_id=chat_message_data.session_id,
                status=Status(
                    indicator=StatusIndicator.ERROR,
                    message="Timeout while fetching the correct party documents",
                ),
            )
            await sio.emit(
                "chat_response_complete",
                chat_response_complete_dto.model_dump(),
                to=sid,
            )
            return

        party_coros = []
        # for party in parties_to_respond:
        logger.info("Comparison response is being fetched by WAHL_CHAT_PARTY")

        party_coros.append(
            fetch_and_emit_party_response(
                sid,
                WAHL_CHAT_PARTY,
                chat_history_str,
                user_message.content,
                chat_session,
                all_available_parties=all_parties,
                use_premium_llms=chat_message_data.user_is_logged_in,
                is_cacheable_chat=chat_session.is_cacheable,
                relevant_docs=relevant_doc_dict,
                parties_being_compared=parties_being_compared,
                is_comparing_question=is_comparing_question,
                improved_rag_query_list=improved_rag_query_list,
            )
        )

    # wait for all coroutines to finish with a timeout
    try:
        await asyncio.wait_for(
            asyncio.gather(*party_coros),
            timeout=40,
        )
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout while fetching party responses: {e}")
        chat_response_complete_dto = ChatResponseCompleteDto(
            session_id=chat_message_data.session_id,
            status=Status(
                indicator=StatusIndicator.ERROR,
                message="Timeout while fetching party responses",
            ),
        )
        await sio.emit(
            "chat_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    # Create a list of the parties that have been selected to respond and the parties that are already in the chat
    ids_of_parties_in_chat = set(pre_selected_party_ids + party_id_list)
    parties_in_chat = [
        party for party in all_parties if party.party_id in ids_of_parties_in_chat
    ]

    full_conversation_history_str = build_chat_history_string(chat_history, all_parties)

    try:
        chat_title_and_quick_replies = await generate_chat_title_and_chick_replies(
            chat_history_str=full_conversation_history_str,
            chat_title=chat_session.title or "Noch kein Titel vergeben",
            parties_in_chat=parties_in_chat,
            wahl_chat_assistant_last_responded=party_id_list
            == [WAHL_CHAT_PARTY.party_id],
            is_comparing=is_comparing_question,
        )
    except openai.BadRequestError as e:
        logger.error(
            f"Error generating chat title and quick replies: {e}", exc_info=True
        )
        chat_response_complete_dto = ChatResponseCompleteDto(
            session_id=chat_message_data.session_id,
            status=Status(
                indicator=StatusIndicator.ERROR,
                message=f"Error generating chat title and quick replies: {e}",
            ),
        )
        await sio.emit(
            "chat_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    quick_replies_and_title_dto = QuickRepliesAndTitleDto(
        session_id=chat_session.session_id,
        quick_replies=chat_title_and_quick_replies.quick_replies,
        title=chat_title_and_quick_replies.chat_title,
    )
    logger.debug(f"Emitting quick replies and title for client {sid}")
    await sio.emit(
        "quick_replies_and_title_ready",
        quick_replies_and_title_dto.model_dump(),
        to=sid,
    )
    chat_session.last_quick_replies = chat_title_and_quick_replies.quick_replies

    chat_response_complete_dto = ChatResponseCompleteDto(
        session_id=chat_session.session_id,
        status=Status(
            indicator=StatusIndicator.SUCCESS,
            message="Success",
        ),
    )
    await sio.emit(
        "chat_response_complete",
        chat_response_complete_dto.model_dump(),
        to=sid,
    )


@sio.on("voting_behavior_request")
async def get_voting_behavior(sid: str, body: dict):
    try:
        improved_rag_query = None
        request_data = VotingBehaviorRequestDto(**body)

        party = await aget_party_by_id(request_data.party_id)

        if party is None:
            raise ValueError(f"Party {request_data.party_id} not found")

        # Improve the RAG query
        improved_rag_query = await get_improved_rag_query_voting_behavior(
            party, request_data.last_user_message, request_data.last_assistant_message
        )

        # Get the relevant votes for the last answer
        relevant_votes = await identify_relevant_votes(improved_rag_query)

        # Collect all votes first
        votes: list[Vote] = []
        for vote_doc in relevant_votes:
            vote_data_json_str = vote_doc.metadata.get("vote_data_json_str", "{}")
            vote_data = json.loads(vote_data_json_str)
            vote = Vote(**vote_data)

            # Check if the party really voted in this vote
            res = [
                party_vote
                for party_vote in vote.voting_results.by_party
                if party_vote.party == party.party_id
            ]
            if not res:
                continue

            votes.append(vote)
            # Emit each vote as it's processed
            vote_dto = VotingBehaviorVoteDto(
                request_id=request_data.request_id,
                vote=vote,
            )
            await sio.emit(
                "voting_behavior_result",
                vote_dto.model_dump(),
                to=sid,
            )

        # Stream the vote behavior summary
        complete_message = ""
        if party:
            summary_stream = await generate_party_vote_behavior_summary(
                party,
                request_data.last_user_message,
                request_data.last_assistant_message,
                votes,
                summary_llm_size=request_data.summary_llm_size,
                use_premium_llms=request_data.user_is_logged_in,
            )
            chunk_index = 0
            async for chunk in summary_stream:
                chunk_content = chunk.content
                if isinstance(chunk_content, str):
                    complete_message += chunk_content

                for i in range(0, len(chunk_content), MAX_RESPONSE_CHUNK_LENGTH):
                    if i > 0:
                        # Sleep for a short time to simulate processing time
                        await asyncio.sleep(0.025)
                    split_chunk_content = chunk_content[
                        i : i + MAX_RESPONSE_CHUNK_LENGTH
                    ]
                    summary_chunk_dto = VotingBehaviorSummaryChunkDto(
                        request_id=request_data.request_id,
                        chunk_index=chunk_index,
                        summary_chunk=split_chunk_content,
                        is_end=False,
                    )

                    # Still emit chunks for progressive display
                    await sio.emit(
                        "voting_behavior_summary_chunk",
                        summary_chunk_dto.model_dump(),
                        to=sid,
                    )
                    chunk_index += 1

        # Send final response
        voting_behavior = VotingBehaviorDto(
            request_id=request_data.request_id,
            message=complete_message,
            status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
            votes=votes,
            rag_query=improved_rag_query,
        )

        await sio.emit("voting_behavior_complete", voting_behavior.model_dump(), to=sid)
    except openai.BadRequestError as e:
        logger.error(f"Error processing voting behavior request: {e}", exc_info=True)
        error_response = VotingBehaviorDto(
            request_id=body.get("request_id"),
            message="Hierzu kann ich leider keine Informationen bereitstellen.",
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
            votes=[],
            rag_query=improved_rag_query,
        )
    except Exception as e:
        logger.error(f"Error processing voting behavior request: {e}", exc_info=True)
        error_response = VotingBehaviorDto(
            request_id=body.get("request_id"),
            message="Es tut mir Leid, es ist ein Fehler aufgetreten. Bitte versuche es später erneut.",
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
            votes=[],
            rag_query=improved_rag_query,
        )
        await sio.emit("voting_behavior_complete", error_response.model_dump(), to=sid)


@sio.on("mock_websocket_usage")
async def mock_websocket_usage(sid: str, body: dict):
    # initialize chat session
    await init_chat_session(sid, body)

    NUMBER_OF_REQUESTS = 10
    NUMBER_OF_COROUTINES = 10
    NUMBER_OF_CHUNKS = 1000

    async def fetch_cat_fact(session):
        """Fetches a single cat fact from the API."""
        url = "https://httpbin.org/get"
        async with session.get(url) as response:
            if response.status == 200:
                return "is_da"
            else:
                return f"Error: {response.status}"

    async def fetch_facts_coroutine(id, num_requests):
        """Coroutine to fetch a number of cat facts and store them in a shared list."""
        # Create a TCPConnector with a larger pool size
        connector = aiohttp.TCPConnector(limit=200)
        async with aiohttp.ClientSession(connector=connector) as session:
            for _ in range(num_requests):
                _ = await fetch_cat_fact(session)

    # test the effect of multiple coroutines
    num_coroutines = NUMBER_OF_COROUTINES
    num_requests_per_coroutine = NUMBER_OF_REQUESTS

    # Create a list of coroutines
    tasks = [
        fetch_facts_coroutine(i, num_requests_per_coroutine)
        for i in range(num_coroutines)
    ]

    # Run all coroutines concurrently
    await asyncio.gather(*tasks)

    # mock emitting of party responses
    logger.info(f"Mock emitting of {NUMBER_OF_CHUNKS} party responses for client {sid}")
    for i in range(NUMBER_OF_CHUNKS):
        await sio.emit("mock_response_chunk_ready", {"message": f"{i}"}, to=sid)
    logger.info(f"Completed: Mock emitting of party responses for client {sid}")

    # mock emitting of party response complete
    await sio.emit("mock_response_complete", {"message": "Success"}, to=sid)


@sio.on("swiper_assistant_session_init")
async def init_swiper_assistant_session(sid: str, body: dict):
    logger.debug(
        f"Client {sid} requested wahl-chat-swiper session initialization with body: {body}"
    )
    try:
        init_chat_session_dto = InitChatSessionDto(**body)
    except ValidationError as e:
        logger.error(
            f"Error validating wahl-chat-swiper session initialization request for client {sid}: {e}"
        )
        chat_session_initialized_dto = ChatSessionInitializedDto(
            session_id=None,
            status=Status(indicator=StatusIndicator.ERROR, message=str(e)),
        )
        await sio.emit(
            "swiper_assistant_session_initialized",
            chat_session_initialized_dto.model_dump(),
            to=sid,
        )
        return

    logger.debug(f"Creating wahl-chat-swiper session: {init_chat_session_dto}")

    chat_session = GroupChatSession(
        session_id=init_chat_session_dto.session_id,
        title=init_chat_session_dto.current_title,
        chat_history=init_chat_session_dto.chat_history,
        chat_response_llm_size=init_chat_session_dto.chat_response_llm_size,
    )

    async with sio.session(sid) as session:
        session["swiper_assistant_sessions"] = session.get(
            "swiper_assistant_sessions", {}
        )
        session["swiper_assistant_sessions"][chat_session.session_id] = chat_session

    logger.debug(f"Chat session initialized for client {sid}")
    chat_session_initialized_dto = ChatSessionInitializedDto(
        session_id=init_chat_session_dto.session_id,
        status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
    )

    await sio.emit(
        "swiper_assistant_session_initialized",
        chat_session_initialized_dto.model_dump(),
        to=sid,
    )


@sio.on("swiper_assistant_answer_request")
async def swiper_assistant_answer_request(sid: str, body: dict):
    logger.debug(f"Client {sid} requested wahl-chat-swiper answer with body: {body}")

    try:
        chat_message_data = WahlChatSwiperUserMessageDto(**body)
    except ValidationError as e:
        logger.error(
            f"Error validating wahl-chat-swiper message data for client {sid}: {e}"
        )
        chat_response_complete_dto = WahlChatSwiperResponseCompleteDto(
            session_id=None,
            complete_message=Message(
                role=Role.ASSISTANT,
                content="Es tut mir Leid, leider ist ein Fehler aufgetreten. Bitte versuche es später erneut.",
                sources=[],
            ),
            status=Status(
                indicator=StatusIndicator.ERROR,
                message=str(e),
            ),
        )
        await sio.emit(
            "swiper_assistant_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    logger.debug(f"wahl.chat Swiper message data: {chat_message_data}")

    # Extract user message
    user_message = Message(
        role="user",
        content=chat_message_data.user_message,
    )

    # Access chat session from socket session
    try:
        async with sio.session(sid) as session:
            chat_session: GroupChatSession = session.get(
                "swiper_assistant_sessions", {}
            ).get(chat_message_data.session_id)

            # Update session with user message
            chat_history = chat_session.chat_history
            # Append the user message if it not identical to the last message
            if (
                len(chat_history) == 0
                or chat_history[-1].content != user_message.content
            ):
                chat_history.append(user_message)
    except Exception as e:
        logger.error(
            f"Error accessing wahl-chat-swiper session for client {sid}: {e}",
            exc_info=True,
        )
        chat_response_complete_dto = WahlChatSwiperResponseCompleteDto(
            session_id=chat_message_data.session_id,
            complete_message=Message(
                role=Role.ASSISTANT,
                content="Es tut mir Leid, leider ist ein Fehler aufgetreten. Bitte versuche es später erneut.",
                sources=[],
            ),
            status=Status(
                indicator=StatusIndicator.ERROR,
                message="It seems like the chat session has not been started",
            ),
        )
        await sio.emit(
            "swiper_assistant_response_complete",
            chat_response_complete_dto.model_dump(),
            to=sid,
        )
        return

    chat_history_without_last_user_message = chat_history[:-1]
    chat_history_str = build_chat_history_string(
        chat_history_without_last_user_message, []
    )

    swiper_assistant_response = await generate_swiper_assistant_response(
        current_political_question=chat_message_data.current_political_question,
        conversation_history=chat_history_str,
        user_message=chat_message_data.user_message,
        chat_response_llm_size=chat_session.chat_response_llm_size,
    )

    chat_session.chat_history.append(swiper_assistant_response)

    chat_response_complete_dto = WahlChatSwiperResponseCompleteDto(
        session_id=chat_message_data.session_id,
        complete_message=swiper_assistant_response,
        status=Status(indicator=StatusIndicator.SUCCESS, message="Success"),
    )
    await sio.emit(
        "swiper_assistant_response_complete",
        chat_response_complete_dto.model_dump(),
        to=sid,
    )
