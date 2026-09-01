"""
STAGE 2C - Qdrant Hybrid Vector Database Ingestion

Purpose
-------
Read Stage 2B chunk JSONL, generate dense + sparse embeddings,
and ingest the chunks into a Qdrant hybrid-search collection.

Requirements
------------
pip install -U qdrant-client fastembed

Start Qdrant locally
--------------------
docker run -p 6333:6333 qdrant/qdrant

Or:

d:
cd D://QdrantDB
qdrant.exe

>curl -X DELETE "http://localhost:6333/collections/hybrid_rag

Input
-----
Stage 2B JSONL chunk file.

Stored in Qdrant
----------------
Dense vector:
    
    jinaai/jina-embeddings-v2-base-en

Sparse vector:
    Qdrant/bm25

Payload:
    document_name
    document_id
    chunk_id
    page_no
    section_path       -> normalized string using " > "
    table_ref
    chunk_text
    retrieval_text

Indexes
-------
document_id -> used to filter retrieval to documents selected by UI
chunk_id    -> useful for direct chunk lookup / debugging

Important
---------
The ingestion is idempotent because the Qdrant point ID is generated
deterministically from document_id + chunk_id.

Therefore, rerunning the script updates the same points rather than
creating duplicate points.
"""

import json
import os
import uuid
from typing import Any

from dotenv import load_dotenv
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models


# ==========================================================
# CONFIGURATION
# ==========================================================

load_dotenv()



QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "hybrid_rag"


JSONL_PATH = r"D:/VECTORDBCREATION/document_processing_db_creation/output_stage_2_chunk_metadata_generation/chunks.jsonl"


BATCH_SIZE = 128

# DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
SPARSE_MODEL_NAME = "Qdrant/bm25"

VECTOR_NAME_DENSE = "text-dense"
VECTOR_NAME_SPARSE = "text-sparse"

DENSE_DIMENSION = 768

# Namespace used to generate deterministic UUIDs for Qdrant points.
POINT_ID_NAMESPACE = uuid.UUID(
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
)


# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def normalize_section_path(section_path: Any) -> str:
    """
    Convert section_path into the canonical string representation.

    Example:

        ["Enterprise Technology Governance Report",
         "Governance Framework"]

    becomes:

        "Enterprise Technology Governance Report > Governance Framework"

    Handles empty/missing section paths safely.
    """

    if not section_path:
        return ""

    if isinstance(section_path, str):
        return section_path.strip()

    if isinstance(section_path, list):
        parts = []

        for part in section_path:
            if part is None:
                continue

            part = str(part).strip()

            if part:
                parts.append(part)

        return " > ".join(parts)

    raise TypeError(
        f"section_path must be a list or string, "
        f"got {type(section_path).__name__}"
    )


def build_retrieval_text(
    section_path: Any,
    chunk_text: Any,
) -> tuple[str, str]:
    """
    Build the canonical section_path string and retrieval_text.

    retrieval_text format:

        Section Path

        Chunk Text
    """

    section_path_string = normalize_section_path(section_path)

    if chunk_text is None:
        chunk_text_string = ""
    else:
        chunk_text_string = str(chunk_text).strip()

    if section_path_string and chunk_text_string:
        retrieval_text = (
            f"{section_path_string}\n\n"
            f"{chunk_text_string}"
        )

    elif section_path_string:
        retrieval_text = section_path_string

    else:
        retrieval_text = chunk_text_string

    return section_path_string, retrieval_text


def make_point_id(
    document_id: str,
    chunk_id: str,
) -> str:
    """
    Generate a deterministic UUID from document_id + chunk_id.

    This makes ingestion idempotent.

    If the same chunk is ingested again, Qdrant updates
    the existing point instead of creating a duplicate.
    """

    unique_key = f"{document_id}::{chunk_id}"

    return str(
        uuid.uuid5(
            POINT_ID_NAMESPACE,
            unique_key,
        )
    )


def validate_record(record: dict, line_number: int) -> None:
    """
    Validate fields required by the ingestion pipeline.
    """

    required_fields = [
        "document_name",
        "document_id",
        "chunk_id",
        "page_no",
        "section_path",
        "table_ref",
        "chunk_text",
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in record
    ]

    if missing_fields:
        raise ValueError(
            f"Line {line_number}: missing required fields: "
            f"{missing_fields}"
        )


# ==========================================================
# CONNECT TO QDRANT
# ==========================================================


if __name__ == '__main__':
    print("Connecting to Qdrant...")

    client = QdrantClient(
        url=QDRANT_URL
        
    )

    # Verify connection before doing any ingestion.
    try:
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(
            f"Could not connect to Qdrant at {QDRANT_URL}. "
            f"Make sure the Qdrant container is running."
        ) from exc

    print("Successfully connected to Qdrant.")


    # ==========================================================
    # LOAD EMBEDDING MODELS
    # ==========================================================

    print("\nLoading embedding models...")

    dense_model = TextEmbedding(
        model_name=DENSE_MODEL_NAME
    )

    sparse_model = SparseTextEmbedding(
        model_name=SPARSE_MODEL_NAME
    )

    print("Embedding models loaded.")


    # ==========================================================
    # CREATE COLLECTION IF REQUIRED
    # ==========================================================

    if client.collection_exists(COLLECTION_NAME):

        print(
            f"\nCollection '{COLLECTION_NAME}' already exists."
        )

    else:

        print(
            f"\nCreating collection '{COLLECTION_NAME}'..."
        )

        client.create_collection(
            collection_name=COLLECTION_NAME,

            vectors_config={
                VECTOR_NAME_DENSE: models.VectorParams(
                    size=DENSE_DIMENSION,
                    distance=models.Distance.COSINE,
                )
            },

            sparse_vectors_config={
                VECTOR_NAME_SPARSE: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

        print("Collection created.")


    # ==========================================================
    # CREATE PAYLOAD INDEXES
    # ==========================================================
    #
    # These indexes are required for efficient metadata filtering.
    #
    # document_id:
    #     Used by retrieval to restrict search to documents selected
    #     by the user in the UI.
    #
    # chunk_id:
    #     Useful for direct lookup/debugging.
    #
    # We execute this regardless of whether the collection was
    # newly created or already existed.
    # ==========================================================

    print("\nEnsuring payload indexes exist...")

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print("Payload indexes ready.")


    # ==========================================================
    # VALIDATE INPUT FILE
    # ==========================================================

    if not os.path.isfile(JSONL_PATH):
        raise FileNotFoundError(
            f"JSONL file not found:\n{JSONL_PATH}"
        )

    print(f"\nReading input file:\n{JSONL_PATH}")


    # ==========================================================
    # INGESTION
    # ==========================================================

    batch_records = []
    batch_texts = []

    total_chunks = 0
    line_number = 0


    with open(
        JSONL_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line_number += 1

            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}"
                ) from exc

            validate_record(
                record,
                line_number,
            )

            # --------------------------------------------------
            # Normalize section_path
            # --------------------------------------------------

            section_path_string, retrieval_text = (
                build_retrieval_text(
                    record["section_path"],
                    record["chunk_text"],
                )
            )

            # Store the normalized section path alongside the
            # original record for payload creation.
            batch_records.append(
                (
                    record,
                    section_path_string,
                    retrieval_text,
                )
            )

            batch_texts.append(retrieval_text)

            # --------------------------------------------------
            # Process full batch
            # --------------------------------------------------

            if len(batch_records) >= BATCH_SIZE:

                print(
                    f"Generating embeddings for batch "
                    f"starting at chunk {total_chunks + 1}..."
                )

                # dense_vectors = list(
                #     dense_model.embed(batch_texts)
                # )

                # Restrict batch slice sizes to prevent RAM allocation crash
                dense_vectors = list(
                    dense_model.embed(
                        batch_texts,
                        batch_size=4,   # Processes 4 chunks at a time sequentially
                        parallel=0      # Disables memory-heavy multithreading
                    )
                )

                # sparse_vectors = list(
                #     sparse_model.embed(batch_texts)
                # )


                sparse_vectors = list(
                    sparse_model.embed(
                        batch_texts,
                        batch_size=4,
                        parallel=0
                    )
                )

                if not (
                    len(dense_vectors)
                    == len(sparse_vectors)
                    == len(batch_records)
                ):
                    raise RuntimeError(
                        "Embedding count does not match "
                        "record count."
                    )

                points = []

                for (
                    rec,
                    section_path_string,
                    retrieval_text,
                ), dense, sparse in zip(
                    batch_records,
                    dense_vectors,
                    sparse_vectors,
                ):

                    point_id = make_point_id(
                        rec["document_id"],
                        rec["chunk_id"],
                    )

                    payload = {
                        "document_name": rec["document_name"],
                        "document_id": rec["document_id"],
                        "chunk_id": rec["chunk_id"],
                        "page_no": rec["page_no"],
                        "section_path": section_path_string,
                        "table_ref": rec["table_ref"],
                        "chunk_text": rec["chunk_text"],
                        "retrieval_text": retrieval_text,
                    }

                    points.append(
                        models.PointStruct(
                            id=point_id,

                            vector={
                                VECTOR_NAME_DENSE:
                                    dense.tolist(),

                                VECTOR_NAME_SPARSE:
                                    models.SparseVector(
                                        indices=sparse.indices.tolist(),
                                        values=sparse.values.tolist(),
                                    ),
                            },

                            payload=payload,
                        )
                    )

                # --------------------------------------------------
                # Upsert batch
                # --------------------------------------------------

                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True,
                )

                total_chunks += len(points)

                print(
                    f"Uploaded {total_chunks} chunks."
                )

                batch_records.clear()
                batch_texts.clear()


    # ==========================================================
    # PROCESS REMAINING RECORDS
    # ==========================================================

    if batch_records:

        print(
            f"Generating embeddings for final batch "
            f"({len(batch_records)} chunks)..."
        )

        # dense_vectors = list(
        #     dense_model.embed(batch_texts)
        # )

        # sparse_vectors = list(
        #     sparse_model.embed(batch_texts)
        # )

        # Restrict batch slice sizes here too for safe final cleanup
        dense_vectors = list(
            dense_model.embed(
                batch_texts,
                batch_size=4,   # Keeps memory steady during final cleanup
                parallel=0
            )
        )

        sparse_vectors = list(
            sparse_model.embed(
                batch_texts,
                batch_size=4,
                parallel=0
            )
        )

        if not (
            len(dense_vectors)
            == len(sparse_vectors)
            == len(batch_records)
        ):
            raise RuntimeError(
                "Embedding count does not match "
                "record count in final batch."
            )

        points = []

        for (
            rec,
            section_path_string,
            retrieval_text,
        ), dense, sparse in zip(
            batch_records,
            dense_vectors,
            sparse_vectors,
        ):

            point_id = make_point_id(
                rec["document_id"],
                rec["chunk_id"],
            )

            payload = {
                "document_name": rec["document_name"],
                "document_id": rec["document_id"],
                "chunk_id": rec["chunk_id"],
                "page_no": rec["page_no"],
                "section_path": section_path_string,
                "table_ref": rec["table_ref"],
                "chunk_text": rec["chunk_text"],
                "retrieval_text": retrieval_text,
            }

            points.append(
                models.PointStruct(
                    id=point_id,

                    vector={
                        VECTOR_NAME_DENSE:
                            dense.tolist(),

                        VECTOR_NAME_SPARSE:
                            models.SparseVector(
                                indices=sparse.indices.tolist(),
                                values=sparse.values.tolist(),
                            ),
                    },

                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        total_chunks += len(points)


    # ==========================================================
    # FINAL VERIFICATION
    # ==========================================================

    collection_info = client.get_collection(
        COLLECTION_NAME
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Chunks processed : {total_chunks}")
    print(f"Dense model      : {DENSE_MODEL_NAME}")
    print(f"Sparse model     : {SPARSE_MODEL_NAME}")
    print(f"Dense dimension  : {DENSE_DIMENSION}")
    print(
        f"Collection points: "
        f"{collection_info.points_count}"
    )

    print("\nPayload fields:")
    print("  - document_name")
    print("  - document_id")
    print("  - chunk_id")
    print("  - page_no")
    print("  - section_path")
    print("  - table_ref")
    print("  - chunk_text")
    print("  - retrieval_text")

    print("\nPayload indexes:")
    print("  - document_id")
    print("  - chunk_id")

    print("\nSUCCESS: Qdrant ingestion is complete.")