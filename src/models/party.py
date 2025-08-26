# SPDX-FileCopyrightText: 2025 2025 wahl.chat
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from pydantic import BaseModel, Field


class Party(BaseModel):
    party_id: str = Field(..., description="The key/id of the party")
    name: str = Field(..., description="The name of the party")
    long_name: str = Field(..., description="The long name of the party")
    description: str = Field(..., description="The description of the party")
    website_url: str = Field(..., description="The website URL of the party")
    candidate: str = Field(..., description="The candidate of the party")
    election_manifesto_url: str = Field(
        ..., description="The URL of the election manifesto of the party"
    )
    is_small_party: bool = Field(
        description="Boolean True, if the party is a small party, otherwise False",
        default=False,
    )
    is_already_in_parliament: bool = Field(
        description="Boolean True, if the party is already in parliament, otherwise False",
        default=True,
    )


WAHL_CHAT_PARTY = Party(
    party_id="wahl-chat",
    name="wahl.chat",
    long_name="wahl.chat Assistent",
    description=(
        "Der wahl.chat Assistent kann allgemeine Fragen zur Bundestagswahl 2025, zum Wahlsystem und zur Anwendung wahl.chat beantworten. "
        "Falls Parteien miteinander verglichen werden, ist er neutral und gibt einen quellenbasierten Überblick."
    ),
    website_url="https://wahl.chat",
    candidate="Wahl Chat",
    election_manifesto_url="https://wahl.chat/presse",
    is_small_party=False,
    is_already_in_parliament=False,
)
