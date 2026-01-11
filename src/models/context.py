# SPDX-FileCopyrightText: 2025 wahl.chat
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from datetime import date as date_type
from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, Field


# =============================================================================
# Constants
# =============================================================================

DEFAULT_CONTEXT_ID = "bundestagswahl-2025"

# =============================================================================
# Hardcoded Reference Data (for easier i18n later)
# =============================================================================


class Country(NamedTuple):
    """Country reference data."""

    code: str
    name: str


class Region(NamedTuple):
    """Region reference data."""

    code: str
    country_code: str
    name: str


# ISO 3166-1 alpha-2 country codes
COUNTRIES: dict[str, Country] = {
    "de": Country(code="de", name="Deutschland"),
    "at": Country(code="at", name="Österreich"),
}


# ISO 3166-2 style region codes (country-region)
REGIONS: dict[str, Region] = {
    # Germany
    "de-bw": Region(code="de-bw", country_code="de", name="Baden-Württemberg"),
    "de-by": Region(code="de-by", country_code="de", name="Bayern"),
    "de-be": Region(code="de-be", country_code="de", name="Berlin"),
    "de-bb": Region(code="de-bb", country_code="de", name="Brandenburg"),
    "de-hb": Region(code="de-hb", country_code="de", name="Bremen"),
    "de-hh": Region(code="de-hh", country_code="de", name="Hamburg"),
    "de-he": Region(code="de-he", country_code="de", name="Hessen"),
    "de-mv": Region(code="de-mv", country_code="de", name="Mecklenburg-Vorpommern"),
    "de-ni": Region(code="de-ni", country_code="de", name="Niedersachsen"),
    "de-nw": Region(code="de-nw", country_code="de", name="Nordrhein-Westfalen"),
    "de-rp": Region(code="de-rp", country_code="de", name="Rheinland-Pfalz"),
    "de-sl": Region(code="de-sl", country_code="de", name="Saarland"),
    "de-sn": Region(code="de-sn", country_code="de", name="Sachsen"),
    "de-st": Region(code="de-st", country_code="de", name="Sachsen-Anhalt"),
    "de-sh": Region(code="de-sh", country_code="de", name="Schleswig-Holstein"),
    "de-th": Region(code="de-th", country_code="de", name="Thüringen"),
}


def get_country(country_code: str) -> Country | None:
    """Get country by code."""
    return COUNTRIES.get(country_code)


def get_region(region_code: str) -> Region | None:
    """Get region by code."""
    return REGIONS.get(region_code)


def get_regions_for_country(country_code: str) -> list[Region]:
    """Get all regions for a country."""
    return [r for r in REGIONS.values() if r.country_code == country_code]


# =============================================================================
# Database Entities
# =============================================================================


class ContextType(str, Enum):
    """Type of political context."""

    ELECTION = "election"
    GENERAL = "general"


class Context(BaseModel):
    """
    A political context representing either an election or a general political level.

    Examples:
    - Elections: "bundestagswahl-2025", "nrw-landtagswahl-2027"
    - General: "bundesebene", "landesebene-bayern"
    """

    context_id: str = Field(..., description="Unique identifier for the context")
    name: str = Field(..., description="Display name (e.g., 'Bundestagswahl 2025')")
    type: ContextType = Field(..., description="Type: election or general")
    date: date_type | None = Field(
        None, description="Relevant date (e.g., election date for elections)"
    )
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    region_code: str | None = Field(
        None, description="ISO 3166-2 region code (null for federal level)"
    )
    is_active: bool = Field(
        True, description="Whether this context is currently active"
    )

    @property
    def country(self) -> Country | None:
        """Get the country for this context."""
        return get_country(self.country_code)

    @property
    def region(self) -> Region | None:
        """Get the region for this context (if any)."""
        if self.region_code:
            return get_region(self.region_code)
        return None

    @property
    def is_federal(self) -> bool:
        """Check if this is a federal-level context."""
        return self.region_code is None


class ContextParty(BaseModel):
    """
    A party within a specific context (stored as sub-collection of Context).

    The same party (e.g., SPD) can have different configurations
    in different contexts (different candidates, manifestos, etc.).

    Firestore path: contexts/{context_id}/parties/{party_id}
    """

    party_id: str = Field(..., description="Party identifier (e.g., 'spd', 'cdu')")
    name: str = Field(..., description="Short name (e.g., 'SPD')")
    long_name: str = Field(
        ..., description="Full name (e.g., 'Sozialdemokratische Partei...')"
    )
    manifesto_url: str | None = Field(
        None, description="URL to the election manifesto/program"
    )
    candidate: str | None = Field(
        None, description="Lead candidate (mainly for elections)"
    )
    website_url: str = Field(..., description="Party website URL")
    is_in_parliament: bool = Field(
        True, description="Whether the party is currently in parliament"
    )
    background_color: str = Field(
        "#808080", description="Brand color for UI (hex format)"
    )
    logo_src: str = Field("", description="URL/path to party logo")
