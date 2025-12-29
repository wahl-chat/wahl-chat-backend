# Multi-Context Support

## Goal

Support multiple political contexts with context-specific parties and documents:
- **Elections**: `bundestagswahl-2025`, `nrw-landtagswahl-2027`, `europawahl-2029`
- **General political levels** (when no election): `bundesebene`, `landesebene-bayern`, `landesebene-nrw`

## Data Model

### Hardcoded Reference Data

Countries and regions are hardcoded constants (not in database) for easier i18n later:

```python
# Use ISO 3166-1 alpha-2 for countries
COUNTRIES = {
    "de": "Deutschland",
    "at": "Österreich",
}

# Use ISO 3166-2:DE style codes for regions
REGIONS = {
    "de-nw": ("de", "Nordrhein-Westfalen"),
    "de-by": ("de", "Bayern"),
    "de-bw": ("de", "Baden-Württemberg"),
    "de-ni": ("de", "Niedersachsen"),
    # ... etc
}
```

### Database Entities

```
Context
├── context_id: string (PK, e.g., "bundestagswahl-2025", "bundesebene")
├── name: string (e.g., "Bundestagswahl 2025", "Bundesebene")
├── type: "election" | "general"
├── date: date | null (e.g., election date)
├── country_code: string (e.g., "de")
├── region_code: string | null (e.g., "de-nw", null for federal)
├── is_active: boolean
└── is_default: boolean (the default context to show)

ContextParty
├── context_party_id: string (PK, e.g., "bundestagswahl-2025_spd")
├── context_id: string (FK -> Context)
├── party_id: string (e.g., "spd", "cdu")
├── long_name: string
├── name: string (e.g., "SPD")
├── manifesto_url: string | null
├── candidate: string | null (only for elections)
├── website_url: string
├── is_in_parliament: boolean
├── background_color: string
└── logo_src: string
```

## Firestore Structure

```
contexts/{context_id}
├── context_id: string
├── name: string
├── type: "election" | "general"
├── date: timestamp | null
├── country_code: string
├── region_code: string | null
├── is_active: boolean
└── is_default: boolean

context_parties/{context_party_id}
├── context_party_id: string (e.g., "bundestagswahl-2025_spd")
├── context_id: string
├── party_id: string
├── long_name: string
├── name: string
├── manifesto_url: string | null
├── candidate: string | null
├── website_url: string
├── is_in_parliament: boolean
├── background_color: string
└── logo_src: string

# Existing collections updated with context_id:
proposed_questions/{context_id}/parties/{party_id}/questions/{question_id}
cached_answers/{context_id}/parties/{party_id}/{cache_key}/{answer_id}
chat_sessions/{session_id}
├── context_id: string  # NEW field
└── ... existing fields
```

## Vector Store

- One Qdrant collection per context: `context_{context_id}_{env}` (e.g., `context_bundestagswahl-2025_prod`)
- Documents indexed with `party_id` metadata for filtering
- Existing `all_parties_{env}` collection migrated to `context_bundestagswahl-2025_{env}`

## API Changes

### Session Start

```python
# Current
POST / session / start
{user_id, party_ids}

# New
POST / session / start
{user_id, context_id, party_ids}
```

### Chat Session Model

```python
class ChatSession:
    context_id: str  # NEW
    user_id: str
    party_id: str
    # ...
```

## Prompt Changes

Replace hardcoded "Bundestagswahl 2025" references with context metadata:

- `{context.name}`
- `{context.date}`

## Document Storage Changes

**Current structure:** `public/{party_id}/{filename}`
**New structure:** `public/{context_id}/{party_id}/{filename}`

### Firebase Function Updates Required

The Cloud Functions in `firebase/functions/main.py` need to be updated:

1. **`is_party_pdf_for_vector_store()`** - Update path validation:
   - Change from 3 parts (`public/{party_id}/{filename}`) to 4 parts (`public/{context_id}/{party_id}/{filename}`)

2. **`on_party_document_upload()`** - Update path parsing:
   - Extract `context_id` from `name.split("/")[1]`
   - Extract `party_id` from `name.split("/")[2]`
   - Extract `filename` from `name.split("/")[3]`
   - Use context-scoped collection: `context_{context_id}_{env}` instead of `all_parties_{env}`
   - Update Firestore path: `sources/{context_id}/{party_id}/source_documents`

3. **`on_party_document_deleted()`** - Same path parsing updates

4. **`add_source_document_to_firebase()` / `delete_source_document_from_firebase()`** - Add `context_id` parameter

## Admin API (follow-up PRs)

### Contexts

```
GET    /admin/contexts
POST   /admin/contexts
GET    /admin/contexts/{context_id}
PUT    /admin/contexts/{context_id}
DELETE /admin/contexts/{context_id}
```

### Context Parties

```
GET    /admin/context-parties?context_id=...
POST   /admin/context-parties                                   { context_id, party_id, ... }
PUT    /admin/context-parties/{context_party_id}
DELETE /admin/context-parties/{context_party_id}
```

### Document Upload (with auto-indexing)

```
GET    /admin/context-parties/{context_party_id}/documents
POST   /admin/context-parties/{context_party_id}/documents
       - Upload PDF/document
       - Stores in Cloud Storage: public/{context_id}/{party_id}/{filename}
       - Indexes into Qdrant collection `context_{context_id}` with party_id metadata

DELETE /admin/context-parties/{context_party_id}/documents/{document_id}
       - Removes from storage
       - Removes from vector index
```

### Auth

Admin endpoints secured via Firebase Admin claims or API key.

## Migration

1. Create Context document for `bundestagswahl-2025`
2. Create ContextParty documents from existing `Party` entries
3. Create Qdrant collection `context_bundestagswahl-2025_{env}`
4. Copy vectors from `all_parties_{env}` to new collection
5. Migrate proposed_questions and cached_answers to new structure

---

## Implementation Plan

### Phase 1: Foundation (No Breaking Changes)

#### PR 1.1: Data Models
**Files:** `src/models/context.py` (new)

- Add `Context` and `ContextParty` Pydantic models
- Add hardcoded `COUNTRIES` and `REGIONS` constants
- No changes to existing code

---

#### PR 1.2: Firestore Structure & Seed Data
**Files:** `firebase/firestore.rules`, `firebase/firestore_data/`

- Add Firestore rules for `contexts` and `context_parties` collections
- Create seed data: `contexts.json`, `context_parties.json`
- Populate with `bundestagswahl-2025` context and parties from existing `parties.json`

---

#### PR 1.3: Firebase Service - Context Methods
**Files:** `src/firebase_service.py`

- `aget_contexts()` → List[Context]
- `aget_context_by_id(context_id)` → Context
- `aget_default_context()` → Context
- `aget_context_parties(context_id)` → List[ContextParty]
- `aget_context_party(context_id, party_id)` → ContextParty

Existing party methods remain unchanged.

---

### Phase 2: Infrastructure

#### PR 2.1: Vector Store - Context-Scoped Collections
**Files:** `src/vector_store_helper.py`

- Add `context_id` parameter to search functions (default: `bundestagswahl-2025`)
- New collection naming: `context_{context_id}_{env}`
- Keep old collection names working temporarily

---

#### PR 2.2: Parameterized Prompts
**Files:** `src/prompts.py`

- Replace hardcoded "Bundestagswahl 2025" → `{context_name}`
- Replace hardcoded date → `{context_date}`
- Add helper: `build_prompt_context(context: Context) -> dict`

---

#### PR 2.3: Firebase Functions - Context-Scoped Storage
**Files:** `firebase/functions/main.py`, `firebase/functions/models.py`

- Update path validation: `public/{context_id}/{party_id}/{filename}` (4 parts)
- Extract `context_id` and `party_id` from storage path
- Use context-scoped Qdrant collection: `context_{context_id}_{env}`
- Update Firestore paths: `sources/{context_id}/{party_id}/source_documents`
- Add `context_id` parameter to `add_source_document_to_firebase()` / `delete_source_document_from_firebase()`

---

### Phase 3: Core API Updates

#### PR 3.1: Session Init with Context
**Files:** `src/websocket_app.py`, `src/models/dtos.py`

- Add optional `context_id` to `chat_session_init` event
- Default to `aget_default_context()` if not provided
- Store `context_id` in session data

---

#### PR 3.2: Chat Service Context Integration
**Files:** `src/chat_service.py`, `src/chatbot_async.py`

- Pass `context_id` through chat flow
- Load `ContextParty` instead of `Party`
- Use context-scoped vector collections
- Inject context metadata into prompts

---

#### PR 3.3: Proposed Questions per Context
**Files:** `src/firebase_service.py`

- Update to: `proposed_questions/{context_id}/parties/{party_id}/questions/...`
- Update `aget_proposed_questions_for_party(context_id, party_id)`

---

#### PR 3.4: Cached Answers per Context
**Files:** `src/firebase_service.py`

- Update to: `cached_answers/{context_id}/parties/{party_id}/...`
- Update cache read/write methods

---

### Phase 4: Migration

#### PR 4.1: Migration Script
**Files:** `scripts/migrate_to_multi_context.py` (new)

Script to:
1. Create `bundestagswahl-2025` context document
2. Create ContextParty documents from Party documents
3. Create new Qdrant collection, copy vectors
4. Migrate proposed_questions structure
5. Verification checks

---

#### PR 4.2: Cleanup Old Paths
**Files:** Various

- Remove hardcoded `PartyID` enum
- Add deprecation warnings to old methods
- Update documentation

---

### Phase 5: Admin API (Follow-up)

#### PR 5.1: Admin Auth Middleware
**Files:** `src/admin/` (new module)

- Admin authentication middleware
- `/admin` route prefix

---

#### PR 5.2: Contexts CRUD

```
GET/POST       /admin/contexts
GET/PUT/DELETE /admin/contexts/{context_id}
```

---

#### PR 5.3: Context Parties CRUD

```
GET/POST       /admin/context-parties
GET/PUT/DELETE /admin/context-parties/{context_party_id}
```

---

#### PR 5.4: Document Upload

```
GET/POST   /admin/context-parties/{context_party_id}/documents
DELETE     /admin/context-parties/{context_party_id}/documents/{doc_id}
```

- Auto-index to Qdrant on upload
- Remove from Qdrant on delete

---

## PR Dependency Graph

```
Phase 1 (can be merged independently):
  PR 1.1 ──┐
  PR 1.2 ──┼── PR 1.3
           │
Phase 2:   │
  PR 2.1 ──┤
  PR 2.2 ──┤
  PR 2.3 ──┤
           │
Phase 3:   │
  PR 3.1 ──┼── PR 3.2
  PR 3.3 ──┤
  PR 3.4 ──┤
           │
Phase 4:   │
  PR 4.1 ──┴── PR 4.2

Phase 5 (independent, can start after Phase 1):
  PR 5.1 ── PR 5.2 ── PR 5.3 ── PR 5.4
```
