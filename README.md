<!--
SPDX-FileCopyrightText: 2025 2025 wahl.chat

SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
-->

# wahl.chat backend
Backend of the leading political information chatbot for the German federal elections 2025.

## About wahl.chat
#### Links
Application URL: https://wahl.chat/ </br>
About page: https://wahl.chat/about-us </br>

#### Our goal in a nutshell
The aim of wahl.chat is to enable users to engage in a contemporary way with the positions of political parties and to receive answers to individual questions that can be substantiated with sources.

#### Contributions welcome
We appreciate contributions from our community. Please take a look at the open issues, if you are interested.
If you are unsure where to start, please contact robin@wahl.chat.
Further specifications coming soon.

## License
This project is **source-available** under the **PolyForm Noncommercial 1.0.0** license.
- Free for **non-commercial** use (see LICENSE for permitted purposes)
- Share the license text and any `Required Notice:` lines when distributing
- Please contact us at info@wahl.chat to
  a. Inform us about your use case
  b. Get access to assets required for a reference to wahl.chat on your project page
- Do not use the wahl.chat name or logo in your project without our permission

## Localization
This project was initially implemented for the German political system.
To adapt it for use in other countries, you will need to adjust the prompts and data schemas to fit the target locale and political context.



## Setup

> **ℹ️ Need help?**
> If you have any remaining setup questions, please contact us at [info@wahl.chat](mailto:info@wahl.chat).


### Install Requirements
1. Install poetry using the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer) using python 3.10-3.12 (`curl -sSL https://install.python-poetry.org | python3 -`)
2. Run `poetry install` in the root directory of this repository (run `poetry install --with dev` to install with dev dependencies)

### Install Pre-commit Hooks
`poetry run pre-commit install`

### Configure Environment Variables
Create a .env file with the environment variables as defined in `.env.example`.
Regarding the Langchain API key you have 3 Options (it is needed for langchain tracing on smith.langchain.com)
1. Set up your own project and API key at [smith.langchain.com](https://smith.langchain.com)
2. Set `LANGCHAIN_TRACING_V2=false` to deactivate tracing.

### Provide Firebase Admin SDK Credentials
#### Option 1: Use gcloud application default credentials
Run `gcloud auth application-default login` and authenticate with your Google Cloud account that has access to the Firebase `wahl-chat-dev` project. Make sure to set the project your Firebase `wahl-chat-dev` project in the gcloud CLI with `gcloud config set project wahl-chat-dev`.

#### Option 2: Use a specific Firebase Admin SDK credentials file
Add a file named `wahl-chat-dev-firebase-adminsdk.json` to the root directory of this repository. This file can be generated at [https://console.firebase.google.com/u/0/project/wahl-chat-dev/settings/serviceaccounts/adminsdk](https://console.firebase.google.com/u/0/project/wahl-chat-dev/settings/serviceaccounts/adminsdk)

## Run
### Locally
`poetry run python -m src.aiohttp_app --debug`

### Docker
1. **Build the image**

   ```bash
   docker build -t wahl-chat:latest .
   ```

2. **Run using a Firebase Admin SDK service account file (dev/prod)**

   - **Prerequisites**:
     - `.env` in the project root (see `.env.example`)
     - For **dev**: `wahl-chat-dev-firebase-adminsdk.json` in the project root **before** building the image
       (download from the Firebase console as described above).
     - For **prod**: `wahl-chat-firebase-adminsdk.json` in the project root **before** building the image.

   - **Run (dev)**:

     ```bash
     docker run --env-file .env -p 8080:8080 wahl-chat:latest
     ```

     With `ENV=dev` in `.env`, the backend will use `wahl-chat-dev-firebase-adminsdk.json` inside the container if present;
     otherwise it falls back to Application Default Credentials.

   - **Run (prod)**:

     ```bash
     docker run --env-file .env -p 8080:8080 wahl-chat:latest
     ```

     With `ENV=prod` in `.env`, the backend will use `wahl-chat-firebase-adminsdk.json` inside the container if present;
     otherwise it falls back to Application Default Credentials.

3. **Run using Google Application Default Credentials (gcloud ADC)**

   - **Prerequisites**:
     - `gcloud auth application-default login` has been run locally.
     - Your ADC file exists at `~/.config/gcloud/application_default_credentials.json`.

   - **Command**:

     ```bash
     ADC=~/.config/gcloud/application_default_credentials.json && \
     docker run --env-file .env \
       -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/keys/application_default_credentials.json \
       -e GOOGLE_CLOUD_PROJECT=wahl-chat-dev \
       -v ${ADC}:/tmp/keys/application_default_credentials.json:ro \
       -p 8080:8080 \
       wahl-chat:latest
     ```

     In this setup, `wahl-chat-*-firebase-adminsdk.json` file is **not** required in the image.
     Firebase Admin will use the mounted ADC credentials instead.
     Make sure to set the `GOOGLE_CLOUD_PROJECT` environment variable to the project ID of the Firebase project you want to use.


## Test
### End-to-end Tests
Prerequisite: run the backend locally

All websocket tests: `poetry run pytest tests/test_websocket_app.py -s`

Specific websocket test: `poetry run pytest tests/test_websocket_app.py -k test_get_chat_answer -s`


## Firebase Management
Note: the following commands are to be run in the `firebase` directory of this repository.

### Prerequisites
Firebase CLI: `npm install -g firebase-tools` & `firebase login`
[node-firestore-import-export](https://github.com/jloosli/node-firestore-import-export/): `npm install -g node-firestore-import-export`

### Deploying Firebase Changes
0. Select the project you want to deploy to: `firebase use dev` or `firebase use prod`
1. Deploying the Firestore Rules: `firebase deploy --only firestore:rules`
2. Deploying the Storage Rules: `firebase deploy --only storage`
3. Deploying the Firestore Indexes: `firebase deploy --only firestore:indexes`
4. Deploying the Firebase Functions: `firebase deploy --only functions`
5. Deploying a specific function: `firebase deploy --only functions:FUNCTION_NAME`

### Managing Data
#### Importing party data from dev to prod
1. Export the parties from the dev-database: `firestore-export --accountCredentials ../wahl-chat-dev-firebase-adminsdk.json --backupFile firestore_data/dev/parties.json --nodePath parties -p`
2. Copy `firestore_data/dev/parties.json` to `firestore_data/prod/parties.json`
3. IMPORTANT: Replace Firebase-Storage dev-URLs with prod-URLs (e.g. https://storage.googleapis.com/wahl-chat-dev.firebasestorage.app/public/bsw/Kurzwahlprogramm%20BTW25_2024-12-21.pdf --> https://storage.googleapis.com/wahl-chat.firebasestorage.app/public/bsw/Kurzwahlprogramm%20BTW25_2024-12-21.pdf): ctrl-f for `https://storage.googleapis.com/wahl-chat-dev.firebasestorage.app` and replace with `https://storage.googleapis.com/wahl-chat.firebasestorage.app`
4. Import the parties to the prod-database: `firestore-import --accountCredentials ../wahl-chat-firebase-adminsdk.json --backupFile firestore_data/prod/parties.json --nodePath parties`

#### Exporting Proposed Questions from one party to another
1. Export from the party where the questions already exist `firestore-export --accountCredentials ../wahl-chat-dev-firebase-adminsdk.json --backupFile firestore_data/proposed_questions_afd_questions.json --nodePath proposed_questions/afd/questions -p`
2. Import to the party where the questions should be added `firestore-import --accountCredentials ../wahl-chat-dev-firebase-adminsdk.json --backupFile firestore_data/proposed_questions_afd_questions.json --nodePath proposed_questions/bsw/questions`

#### Modifying Proposed Questions
1. Export the proposed_questions collection from dev: `firestore-export --accountCredentials ../wahl-chat-dev-firebase-adminsdk.json --backupFile firestore_data/proposed_questions.json --nodePath proposed_questions -p`
2. Modify the `firestore_data/proposed_questions.json` file
3. Import the proposed_questions collection to dev: `firestore-import --accountCredentials ../wahl-chat-dev-firebase-adminsdk.json --backupFile firestore_data/proposed_questions.json --nodePath proposed_questions`

#### Moving Proposed Questions from dev to prod
1. Export the proposed_questions collection from dev: `firestore-export --accountCredentials ../wahl-chat-dev-firebase-adminsdk.json --backupFile firestore_data/proposed_questions.json --nodePath proposed_questions -p`
2. Import the proposed_questions collection to prod: `firestore-import --accountCredentials ../wahl-chat-firebase-adminsdk.json --backupFile firestore_data/proposed_questions.json --nodePath proposed_questions`

### Seeding Contexts, Parties, and Proposed Questions

The application supports multiple election contexts (e.g., Bundestagswahl 2025, Landtagswahl Baden-Württemberg 2026). Each context has its own set of parties and proposed questions stored as sub-collections.

#### Data Structure

```
firebase/firestore_data/dev/
├── contexts.json                                         # All contexts
├── parties_bundestagswahl-2025.json                      # Parties for BTW 2025
├── parties_landtagswahl-baden-wuerttemberg-2026.json     # Parties for LTW BW 2026
├── parties_kommunalwahl-muenchen-2026.json               # Parties for KW München 2026
├── proposed_questions_bundestagswahl-2025.json           # Proposed questions for BTW 2025
├── proposed_questions_landtagswahl-baden-wuerttemberg-2026.json
└── proposed_questions_kommunalwahl-muenchen-2026.json
```

#### Firestore Structure

```
contexts/{context_id}
├── context_id, name, type, date, ...
├── parties/{party_id}
│   └── party_id, name, long_name, manifesto_url, candidate, ...
└── proposed_questions/{party_id}/questions/{question_id}
    └── content, ...
```

#### Seeding with the Python Script (Recommended)

Run from the project root:

```bash
python scripts/seed_firestore.py
```

This script:
1. Reads `contexts.json` and imports all contexts
2. Finds all `parties_{context_id}.json` files and imports them into `contexts/{context_id}/parties`
3. Finds all `proposed_questions_{context_id}.json` files and imports them into `contexts/{context_id}/proposed_questions/{party_id}/questions`

For production:
```bash
ENV=prod python scripts/seed_firestore.py
```

#### Seeding with firestore-import (Manual)

```bash
# Seed contexts
firestore-import -a wahl-chat-dev-firebase-adminsdk.json -n contexts -b firebase/firestore_data/dev/contexts.json -y

# Seed parties for a specific context
firestore-import -a wahl-chat-dev-firebase-adminsdk.json -n contexts/bundestagswahl-2025/parties -b firebase/firestore_data/dev/parties_bundestagswahl-2025.json -y
```

#### Adding a New Context

1. Add the context to `firebase/firestore_data/dev/contexts.json`
2. Create `firebase/firestore_data/dev/parties_{context_id}.json` with the parties
3. Create `firebase/firestore_data/dev/proposed_questions_{context_id}.json` with the proposed questions
4. Run `python scripts/seed_firestore.py`

### Moving a Context from Dev to Prod

When you've finished setting up and testing a context in dev, follow these steps to deploy it to production:

#### 1. Prepare the Data Files

Copy the dev data files to prod and update URLs:

```bash
cd firebase/firestore_data

# Copy contexts (merge with existing prod contexts if needed)
cp dev/contexts.json prod/contexts.json

# Copy parties for the specific context
cp dev/parties_{context_id}.json prod/parties_{context_id}.json

# Copy proposed questions for the specific context
cp dev/proposed_questions_{context_id}.json prod/proposed_questions_{context_id}.json
```

#### 2. Update Firebase Storage URLs

**IMPORTANT**: Replace dev storage URLs with prod URLs in all copied files:

- Find: `https://storage.googleapis.com/wahl-chat-dev.firebasestorage.app`
- Replace with: `https://storage.googleapis.com/wahl-chat.firebasestorage.app`

This applies to party logos, manifesto PDFs, and any other assets stored in Firebase Storage.

#### 3. Upload Assets to Prod Storage

Ensure all referenced assets (logos, PDFs, etc.) exist in the prod Firebase Storage bucket under the same paths as in dev.

#### 4. Seed Production Firestore

```bash
ENV=prod python scripts/seed_firestore.py
```

#### 5. Deploy Qdrant Vector Store Data

The vector store collections are context-specific. Make sure to populate the prod vector store collection for the new context:
- Dev collection: `context_{context_id}_party_docs_dev`
- Prod collection: `context_{context_id}_party_docs` (no suffix)

#### 6. Verify the Deployment

1. Check the Firebase Console to confirm data was imported correctly
2. Test the context in the production application
3. Verify party responses and proposed questions work as expected
