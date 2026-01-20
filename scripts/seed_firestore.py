#!/usr/bin/env python3
"""
Seed Firestore with contexts and parties data.

Usage:
    python scripts/seed_firestore.py

This script:
1. Imports all contexts from firebase/firestore_data/dev/contexts.json
2. Imports parties from firebase/firestore_data/dev/parties_{context_id}.json
   into contexts/{context_id}/parties sub-collection

File naming convention:
- contexts.json: Contains all context documents
- parties_{context_id}.json: Contains parties for a specific context
  Example: parties_bundestagswahl-2025.json -> contexts/bundestagswahl-2025/parties/
"""

import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore

# Configuration
ENV = os.getenv("ENV", "dev")
DATA_DIR = Path(__file__).parent.parent / "firebase" / "firestore_data" / ENV

# Credentials file path
CREDENTIALS_FILE = (
    "wahl-chat-firebase-adminsdk.json"
    if ENV == "prod"
    else "wahl-chat-dev-firebase-adminsdk.json"
)


def initialize_firebase():
    """Initialize Firebase Admin SDK."""
    cred_path = Path(CREDENTIALS_FILE)
    if not cred_path.exists():
        # Try looking in project root
        cred_path = Path(__file__).parent.parent / CREDENTIALS_FILE

    if cred_path.exists():
        cred = credentials.Certificate(str(cred_path))
        firebase_admin.initialize_app(cred)
    else:
        # Use application default credentials
        firebase_admin.initialize_app()

    return firestore.client()


def seed_contexts(db):
    """Seed the contexts collection."""
    contexts_file = DATA_DIR / "contexts.json"

    if not contexts_file.exists():
        print(f"⚠️  Contexts file not found: {contexts_file}")
        return []

    with open(contexts_file) as f:
        contexts = json.load(f)

    print(f"\n📁 Seeding {len(contexts)} contexts...")
    print("-" * 60)

    context_ids = []
    for context_id, context_data in contexts.items():
        print(f"  ✅ {context_id}")
        db.collection("contexts").document(context_id).set(context_data)
        context_ids.append(context_id)

    print(f"\nContexts seeded: {len(context_ids)}")
    return context_ids


def seed_parties(db):
    """Seed parties sub-collections for each context."""
    # Find all party files matching pattern: parties_{context_id}.json
    party_files = list(DATA_DIR.glob("parties_*.json"))

    if not party_files:
        print("\n⚠️  No party files found")
        return

    print(f"\n📁 Found {len(party_files)} party files")
    print("-" * 60)

    total_parties = 0
    for party_file in sorted(party_files):
        # Extract context_id from filename: parties_{context_id}.json
        context_id = party_file.stem.replace("parties_", "")

        with open(party_file) as f:
            parties = json.load(f)

        print(f"\n  📂 {context_id} ({len(parties)} parties)")

        for party_id, party_data in parties.items():
            doc_ref = (
                db.collection("contexts")
                .document(context_id)
                .collection("parties")
                .document(party_id)
            )
            doc_ref.set(party_data)
            print(f"    ✅ {party_id}")
            total_parties += 1

    print(f"\nTotal parties seeded: {total_parties}")


def main():
    print("=" * 60)
    print("Firestore Seed Script")
    print("=" * 60)
    print(f"Environment: {ENV}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Credentials: {CREDENTIALS_FILE}")

    if not DATA_DIR.exists():
        print(f"\n❌ Data directory not found: {DATA_DIR}")
        return

    db = initialize_firebase()

    # Seed contexts first
    seed_contexts(db)

    # Seed parties for each context
    seed_parties(db)

    print("\n" + "=" * 60)
    print("✅ Seeding complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
