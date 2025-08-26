# SPDX-FileCopyrightText: 2025 2025 wahl.chat
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from datetime import datetime
import os
import time
import tempfile
from uuid import uuid4

from firebase_functions.params import StringParam
from firebase_functions.options import SupportedRegion, MemoryOption
from firebase_functions import storage_fn, logger
from firebase_admin import initialize_app, storage, firestore
import google.cloud.firestore

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from models import PartySource  # type: ignore


ENV = StringParam("ENV")  # "dev" or "prod"
OPENAI_API_KEY = StringParam("OPENAI_API_KEY")
PINECONE_API_KEY = StringParam("PINECONE_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_SIZE = 3072  # Embedding sizes for the OpenAI models: https://platform.openai.com/docs/guides/embeddings#how-to-get-embeddings

ALL_PARTIES_INDEX = "all-parties-index"

STORAGE_TRIGGER_FN_REGION = (
    SupportedRegion.EUROPE_WEST1 if ENV.value == "dev" else SupportedRegion.US_EAST1
)

initialize_app()


def is_party_pdf_for_vector_store(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData], name: str
):
    # Make sure the file is in the public subdirectory
    if not name.startswith("public/"):
        logger.info(f"Skipping file as it is not in the public directory: {name}")
        return False

    # Check if the file is in a party's directory
    if len(name.split("/")) < 3:
        logger.info(
            f"Skipping file as it is not in a public subdirectory for a party: {name}"
        )
        return False

    # Check if the file is a PDF
    if not event.data.content_type == "application/pdf":
        # TODO: consider adding support for other document types
        logger.info(f"Skipping file as it is not a PDF: {name}")
        return False
    return True


def download_pdf(bucket_name: str, name: str):
    bucket = storage.bucket(bucket_name)
    pdf_blob = bucket.blob(name)

    # Create a named temporary file for the PDF, ensuring delete=False
    # so we can manage cleanup manually.
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_file_name = tmp_file.name
    tmp_file.close()  # Close the file so we can write to it

    pdf_blob.download_to_filename(tmp_file_name)
    logger.info(f"Downloaded file to temporary path: {tmp_file_name}")

    return tmp_file_name, pdf_blob


def split_pdf(file_path: str):
    # Load the document as a PDF and split it into chunks
    # TODO: consider switching to PDFMiner (https://www.reddit.com/r/LangChain/comments/13jd9wo/comment/jkh2f9j/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    chunk_size = 1000
    chunk_overlap = 100
    length_function = len
    is_separator_regex = False
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
    )
    logger.info(
        f"Splitting the document with chunk size={chunk_size} and overlap={chunk_overlap}"
    )

    splits = text_splitter.split_documents(pages)
    # Free up memory
    del pages
    del loader
    logger.info(f"Split the document into {len(splits)} chunks")
    return splits


def create_index_if_not_exists(pc: Pinecone, index_name: str, embedding_size: int):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Creating Pinecone index {index_name}")
        pc.create_index(
            name=index_name,
            dimension=embedding_size,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="eu-west-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        logger.info(f"Index {index_name} created")


def add_to_index(splits: list[Document], index_name: str, namespace: str):
    logger.info(
        f"Adding {len(splits)} splits to index {index_name} in namespace {namespace}"
    )

    # TODO: consider switching to NV-Embed-v2 or a similar high-performing embedding model (https://huggingface.co/spaces/mteb/leaderboard) potentially by leveraging (https://github.com/michaelfeil/infinity)
    embed = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY.value)

    pc = Pinecone(pinecone_api_key=PINECONE_API_KEY.value, embedding=embed)

    create_index_if_not_exists(pc, index_name, EMBEDDING_SIZE)

    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embed)
    vector_store.add_documents(splits, namespace=namespace)
    logger.info(f"Added splits to index {index_name} in namespace {namespace}")


def add_source_document_to_firebase(
    document_id: str, party_id: str, source: PartySource
):
    firestore_client: google.cloud.firestore.Client = firestore.client()
    source_info_ref = firestore_client.collection(
        f"sources/{party_id}/source_documents"
    ).document(document_id)

    source_info_ref.set(source.model_dump())


def delete_source_document_from_firebase(document_id: str, party_id: str):
    firestore_client: google.cloud.firestore.Client = firestore.client()
    source_info_ref = firestore_client.collection(
        f"sources/{party_id}/source_documents"
    ).document(document_id)
    source_info_ref.delete()


def build_vector_prefix(name: str):
    prefix = name.lower()
    prefix = prefix.replace("/", "#").replace(".", "_")
    # replace umlauts with their ASCII representation
    prefix = prefix.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    return prefix


# Implement a function that adds a pdf document to a vector store when a new document is uploaded to the storage bucket
@storage_fn.on_object_finalized(
    region=STORAGE_TRIGGER_FN_REGION, timeout_sec=540, memory=MemoryOption.GB_1
)
def on_party_document_upload(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],
):
    bucket_name = event.data.bucket
    name = event.data.name
    logger.info(f"Processing file: gs://{bucket_name}/{name}")

    # Print event info
    logger.info(f"Event id: {event.id}")
    logger.info(f"Event type: {event.type}")
    logger.info(f"Event source: {event.source}")
    logger.info(f"Event time: {event.time}")
    logger.info(f"Event data: {event.data}")
    logger.info(f"Event data content type: {event.data.content_type}")
    logger.info(f"Event data size: {event.data.size}")

    if not is_party_pdf_for_vector_store(event, name):
        return

    # Download the document from the storage bucket
    file_path, pdf_blob = download_pdf(bucket_name, name)

    # Split the document into chunks
    splits = split_pdf(file_path)

    # Delete the local file
    os.remove(file_path)

    # Add relevant metadata to the splits
    file_name = name.split("/")[2].replace(".pdf", "")
    file_name_parts = file_name.split("_")
    # Make sure the file name is in the expected format
    if len(file_name_parts) != 2:
        raise ValueError(
            f"File name {file_name} does not match the expected format: [document_name]_[document_date]"
        )
    document_name = file_name_parts[0]
    document_date_str = file_name_parts[1]
    # Make sure the date is in the expected format
    try:
        document_date = datetime.strptime(document_date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"Document date {document_date_str} does not match the expected format: YYYY-MM-DD"
        )

    # Enable and create a public URL for the PDF
    pdf_blob.make_public()
    download_url = pdf_blob.public_url

    prefix = build_vector_prefix(name)
    for split in splits:
        # Add id prefix to the split (https://docs.pinecone.io/guides/data/manage-rag-documents#use-id-prefixes)
        # Rationale: Serverless indices do not support deletion be metadata (https://docs.pinecone.io/guides/data/delete-data#delete-records-by-metadata) but support deletion by id prefix (https://docs.pinecone.io/guides/data/manage-rag-documents#delete-all-records-for-a-parent-document)
        uuid = uuid4()
        split.id = f"{prefix}#{uuid}"

        # Add relevant metadata to the split
        split.metadata["source_document"] = name
        split.metadata["url"] = download_url
        split.metadata["file_name"] = file_name
        split.metadata["document_name"] = document_name
        split.metadata["document_publish_date"] = document_date_str
        if "rede" in prefix:
            split.page_content = f"Ausschnitt aus {prefix}\n\n{split.page_content}"

    # Add the document to the index in the namespace of the party or the general one
    party_subdir = name.split("/")[1]

    add_to_index(splits, ALL_PARTIES_INDEX, namespace=party_subdir)

    # Add the source information to Firestore
    logger.info(
        f"Adding source document {document_name} for party {party_subdir} to Firestore"
    )
    # create datetime object from string
    source = PartySource(
        name=document_name, publish_date=document_date, storage_url=download_url
    )

    add_source_document_to_firebase(
        document_id=file_name, party_id=party_subdir, source=source
    )
    logger.info("Added source information to Firestore")


@storage_fn.on_object_deleted(
    region=STORAGE_TRIGGER_FN_REGION, timeout_sec=540, memory=MemoryOption.MB_512
)
def on_party_document_deleted(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],
):
    bucket_name = event.data.bucket
    name = event.data.name
    logger.info(f"Deleting splits associated with file: gs://{bucket_name}/{name}")

    # Print event info
    logger.info(f"Event id: {event.id}")
    logger.info(f"Event type: {event.type}")
    logger.info(f"Event source: {event.source}")
    logger.info(f"Event time: {event.time}")
    logger.info(f"Event data: {event.data}")

    if not is_party_pdf_for_vector_store(event, name):
        return

    # Extract the namespace from the file path
    party_subdir = name.split("/")[1]

    # Delete source document from Firestore
    file_name = name.split("/")[2].replace(".pdf", "")
    logger.info(f"Deleting source document {file_name} from Firestore")
    delete_source_document_from_firebase(document_id=file_name, party_id=party_subdir)
    logger.info(f"Deleted source document {file_name} from Firestore")

    # Initialize Pinecone and embeddings
    embed = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY.value)
    pc = Pinecone(pinecone_api_key=PINECONE_API_KEY.value, embedding=embed)

    # Define the index name and namespace
    index_name = ALL_PARTIES_INDEX
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(
            f"Index {index_name} does not exist. No deletion of document splits required."
        )
        return

    index = pc.Index(index_name)
    prefix = f"{build_vector_prefix(name)}#"

    logger.info(
        f"Deleting splits from index {index_name} in namespace {party_subdir} with prefix {prefix}"
    )

    # Perform the deletion
    for ids in index.list(prefix=prefix, namespace=party_subdir):
        try:
            logger.info(f"Deleting splits with ids: {ids}")
            index.delete(ids, namespace=party_subdir)
        except Exception as e:
            logger.error(f"Error deleting splits: {e}")

    logger.info(
        f"Deleted splits from index {index_name} in namespace {party_subdir} with prefix {prefix}"
    )
