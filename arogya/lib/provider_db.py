"""Linked network-provider database: contracted package and room rates per hospital.
Reads a synced file here; a live system would call each provider's rate API and keep the sync time."""
import json
from pathlib import Path

DB = json.loads((Path(__file__).parent.parent / "data" / "provider_rates.json").read_text())
ROOM_LABEL = {"general_ward": "General ward", "semi_private": "Semi-private", "private": "Private", "deluxe": "Deluxe"}


def provider(hospital: str) -> dict | None: # return the provider record for a hospital, or None if not found
    return DB["hospitals"].get(hospital)


def package_rate(hospital: str, procedure: str) -> int | None: # return the package rate for a procedure at a hospital, or None if not found
    p = provider(hospital)
    return p["packages"].get(procedure) if p else None


def room_rate(hospital: str, category: str) -> int | None: # return the room rate for a category at a hospital, or None if not found    
    p = provider(hospital)
    return p["rooms"].get(category) if p else None


def synced_at() -> str: # return the timestamp of the last sync of the provider database
    return DB["syncedAt"]
