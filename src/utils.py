# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from pathlib import Path
from typing import Optional, Union
import re
import logging

from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.documents import Document

from openai.types.chat import ChatCompletion
import xxhash

from src.models.chat import Message, Role
from src.models.party import WAHL_CHAT_PARTY, Party

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
EXPECTED_API_NAME = "wahl-chat-api"

logger = logging.getLogger(__name__)


def load_env():
    # Build the path to the .env file
    env_path = BASE_DIR / ".env"
    # Load the environment variables from the .env file
    if not os.getenv("API_NAME") == EXPECTED_API_NAME:
        # The environment variables are not loaded yet
        print(f"Loading environment variables from {env_path}...")
        load_dotenv(env_path, override=True)
        print(f"Loaded environment variables from {env_path}.")
        if not os.getenv("API_NAME"):
            raise ValueError(
                "API_NAME environment variable not set. Please check the contents of your .env file."
            )
        if not os.getenv("API_NAME") == EXPECTED_API_NAME:
            raise ValueError(
                "API_NAME environment variable not set as expected. Please check the contents of your .env file."
            )


def safe_load_api_key(api_key: str) -> Optional[SecretStr]:
    key = os.getenv(api_key)
    if not key:
        return None
    return SecretStr(key)


def get_cors_allowed_origins(env: Optional[str]) -> Union[str, list[str]]:
    if env == "dev":
        return "*"
    else:
        return [
            "https://wahl.chat",
            "https://embed.wahl.chat",
            "https://pre-prod.wahl.chat",
            "https://dev.wahl.chat",
            "http://localhost:3000",
            "http://localhost:8080",
        ]


def build_chat_history_string(
    chat_history: list[Message],
    parties: list[Party],
    default_assistant_name=WAHL_CHAT_PARTY.name,
) -> str:
    chat_history_string = ""
    for i, message in enumerate(chat_history):
        sender = ""
        if message.role == Role.USER:
            sender = "Nutzer"
        else:
            sending_party = next(
                (party for party in parties if party.party_id == message.party_id),
                None,
            )
            if sending_party:
                sender = sending_party.name
            else:
                sender = default_assistant_name
        chat_history_string += f'{i + 1}. {sender}: "{message.content}"\n'
    return chat_history_string


def build_document_string_for_context(
    doc_num: int, doc: Document, doc_num_label="ID"
) -> str:
    return f"""{doc_num_label}: {doc_num}
- Dokumentname: {doc.metadata.get("document_name", "unbekannt")}
- Veröffentlichungsdatum: {doc.metadata.get("document_publish_date", "unbekannt")}
- Inhalt: "{doc.page_content}"

"""


def build_party_str(party: Party):
    return f"""ID: {party.party_id}
- Abkürzung: {party.name}
- Langform: {party.long_name}
- Beschreibung: {party.description}
- Spitzenkandidat*In für die Bundestagswahl 2025: {party.candidate}
- Ist im aktuellen Bundestag vertreten: {party.is_already_in_parliament}
"""


def build_message_from_perplexity_response(response: ChatCompletion) -> Message:
    logger.debug(f"Processing raw perplexity response: {response}")
    # construct a source dict from response citations
    sources = []
    # type ignore because citations actually exists but is not typed
    for link in response.citations:  # type: ignore
        sources.append({"source": link})

    # postprocess perplexity response
    response_text = response.choices[0].message.content

    # give sources addition space before "[id]" -> " [id-1]" or "[id_1, id_2, ...]" --> " [id_1 - 1, id_2 - 1, ...]"
    def replacement(match):
        source_numbers = match.group(1).replace(", ", ",").split(",")
        new_ids = [int(num) - 1 for num in source_numbers]
        new_ids_str = ", ".join([str(num) for num in new_ids])
        # Return the modified string with a space in front
        return f" [{new_ids_str}]"

    # Match patterns like [1], [3], [5] or [1, 2, 3]
    sources_pattern = r"\[((\d+|(\d+, ))*)\]"
    response_text = re.sub(sources_pattern, replacement, response_text or "")
    logger.debug(f"Processed perplexity response text: {response_text}")

    return Message(role=Role.ASSISTANT, content=response_text, sources=sources)


def sanitize_references(text: str) -> str:
    # GPT 4o-mini sometimes references with [id1], [<1>], ... instead of [1]
    # This function sanitizes the references to [1], [2], ... by removing any non-numeric characters from the reference

    def sanitize_citatoin(match):
        content = match.group(1)
        cleaned_content = re.sub(r"[^0-9, ]", "", content)
        return f"[{cleaned_content}]"

    citations_pattern = r"\[(.*?)\]"

    sanitized_text = re.sub(citations_pattern, sanitize_citatoin, text)
    return sanitized_text


if __name__ == "__main__":
    text = """Die Grünen setzen sich für **gute Arbeit** und **faire Löhne** für Fabrikarbeiter ein. Sie wollen:

- **Faire Mindestlöhne**: Ein Mindestlohn von zunächst **15 Euro** im Jahr 2025, der auch für unter 18-Jährige gilt, um die Inflation auszugleichen. [id1]
- **Stärkung der Mitbestimmung**: Die betriebliche Mitbestimmung soll gestärkt werden, um Beschäftigten mehr Einfluss auf ihre Arbeitsbedingungen zu geben. [<2>]
- **Schutz vor Missbrauch**: Gegen Schein-Selbstständigkeit und den Missbrauch von Werkverträgen soll entschieden vorgegangen werden. [id2, id3]

Diese Maßnahmen zielen darauf ab, die Arbeitsbedingungen und die soziale Absicherung für Fabrikarbeiter zu verbessern.
"""
    sanitized_text = sanitize_references(text)
    print(sanitized_text)


def get_chat_history_hash_key(conversation_history_str: str) -> str:
    return xxhash.xxh64(conversation_history_str).hexdigest()
