# SPDX-FileCopyrightText: 2025 2025 wahl.chat
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from enum import StrEnum
from pydantic import BaseModel, Field


class RAG(BaseModel):
    """Output of the RAG Chain."""

    chat_answer: str = Field(
        description="Deine kurze Antwort auf die Nutzerfrage im Markdown-Format mit Hervorhebungen und Absätzen."
    )
    chat_title: str = Field(
        description="Der kurze Titel des Chats in Plain Text. Er soll den Chat kurz und prägnant in 3-5 Worten beschreiben."
    )


class QuickReplyGenerator(BaseModel):
    """Output of the Quick Reply Generator."""

    quick_replies: list[str] = Field(
        description="Liste der drei Quick Replies als Strings."
    )


class PartyID(StrEnum):
    AFD = "afd"
    BSW = "bsw"
    CDU = "cdu"
    FDP = "fdp"
    FREIE_WAEHLER = "fw"
    GRUENE = "gruene"
    LINKE = "linke"
    PIRATEN = "piraten"
    SPD = "spd"
    VOLT = "volt"
    OEDP = "oedp"
    TIERSCHUTZPARTEI = "tierschutzpartei"
    WAHL_CHAT = "wahl-chat"


class PartyListGenerator(BaseModel):
    """Output of the Party List Generator."""

    party_id_list: list[PartyID] = Field(
        description="Liste der Partei-IDs von denen der Nutzer eine Antwort haben will."
    )


class QuestionTypeClassifier(BaseModel):
    """Output of the Question Type Classifier."""

    non_party_specific_question: str = Field(
        description="Die Frage die der Nutzer gestellt hat, jedoch in einer Formulierung als ob sie direkt an eine Partei gerichtet ist."
    )
    is_comparing_question: bool = Field(
        description="True, wenn eine explizite Vergleichsfrage, sonst False."
    )


class ChatSummaryGenerator(BaseModel):
    """Output of the Chat Summary Generator."""

    chat_summary: str = Field(
        description="Die wichtigsten Leitfragen die von den Parteien beantwortet wurden."
    )


class GroupChatTitleQuickReplyGenerator(BaseModel):
    """Output of the Chat Title & Quick Reply Generator."""

    chat_title: str = Field(
        description="Ein kurzer Titel, der den Chat kurz und prägnant in 3-5 Worten beschreibt."
    )
    quick_replies: list[str] = Field(
        description="Liste der drei Quick Replies als Strings."
    )


class RerankingOutput(BaseModel):
    """Output of the Reranking Model."""

    reranked_doc_indices: list[int] = Field(
        description="Absteigend nach Nützlichkeit sortierte Liste der Indices der Dokumente"
    )
