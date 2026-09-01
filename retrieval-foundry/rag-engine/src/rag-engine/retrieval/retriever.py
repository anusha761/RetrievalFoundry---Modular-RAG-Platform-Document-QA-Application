"""
QDRANT HYBRID SEARCH - RRF RETRIEVAL


Purpose
-------
Retrieve relevant chunks from Qdrant using:

    Dense semantic search
        +
    Sparse BM25 keyword search
        ↓
    Reciprocal Rank Fusion (RRF)

This module intentionally does NOT perform:

    - Cross-encoder reranking
    - Table resolution
    - LLM calls
    - FastAPI handling

Initialization
--------------
Qdrant client and embedding models are initialized explicitly
through initialize_retriever().

This prevents expensive model loading on every API request.
"""

from typing import Any, Optional

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models


# ==========================================================
# CONFIGURATION
# ==========================================================

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "hybrid_rag"

DENSE_MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
SPARSE_MODEL_NAME = "Qdrant/bm25"

VECTOR_NAME_DENSE = "text-dense"
VECTOR_NAME_SPARSE = "text-sparse"


# ==========================================================
# RUNTIME RESOURCES
# ==========================================================

qdrant_client: Optional[QdrantClient] = None

dense_model: Optional[TextEmbedding] = None

sparse_model: Optional[SparseTextEmbedding] = None


# ==========================================================
# INITIALIZATION
# ==========================================================

def initialize_retriever() -> None:
    """
    Initialize Qdrant client and embedding models.

    This function should be called ONCE during application
    startup.

    Repeated calls are ignored if resources are already loaded.
    """

    global qdrant_client
    global dense_model
    global sparse_model

    # ------------------------------------------------------
    # Already initialized
    # ------------------------------------------------------

    if (
        qdrant_client is not None
        and dense_model is not None
        and sparse_model is not None
    ):
        return

    # ------------------------------------------------------
    # Connect to Qdrant
    # ------------------------------------------------------

    if qdrant_client is None:

        print("Connecting to Qdrant...")

        qdrant_client = QdrantClient(
            url=QDRANT_URL
        )

        print("Qdrant connection established.")

    # ------------------------------------------------------
    # Load embedding models
    # ------------------------------------------------------

    if dense_model is None or sparse_model is None:

        print("Loading embedding models...")

        if dense_model is None:

            dense_model = TextEmbedding(
                model_name=DENSE_MODEL_NAME
            )

        if sparse_model is None:

            sparse_model = SparseTextEmbedding(
                model_name=SPARSE_MODEL_NAME
            )

        print("Embedding models loaded.")


# ==========================================================
# RESOURCE VALIDATION
# ==========================================================

def _ensure_initialized() -> None:
    """
    Ensure retriever resources have been initialized.
    """

    if (
        qdrant_client is None
        or dense_model is None
        or sparse_model is None
    ):
        raise RuntimeError(
            "Retriever resources are not initialized. "
            "Call initialize_retriever() during application startup."
        )


# ==========================================================
# HYBRID SEARCH FUNCTION
# ==========================================================

def hybrid_search_rrf(
    user_query: str,
    file_id: str,
    candidate_k: int,
) -> list[Any]:
    """
    Perform filtered hybrid search using Qdrant.

    Parameters
    ----------
    user_query : str
        Natural-language query from the user.

    file_id : str
        document_id of the document that should be searched.

    candidate_k : int
        Number of candidates retrieved independently from
        dense and sparse search AND number of results retained
        after RRF.

    Returns
    -------
    list
        Qdrant ScoredPoint objects containing payload and
        the RRF score.
    """

    # ------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError(
            "user_query must be a non-empty string."
        )

    if not isinstance(file_id, str) or not file_id.strip():
        raise ValueError(
            "file_id must be a non-empty string."
        )

    if not isinstance(candidate_k, int) or candidate_k <= 0:
        raise ValueError(
            "candidate_k must be a positive integer."
        )

    # ------------------------------------------------------
    # Ensure resources exist
    # ------------------------------------------------------

    _ensure_initialized()

    user_query = user_query.strip()
    file_id = file_id.strip()

    # ------------------------------------------------------
    # Generate dense query embedding
    # ------------------------------------------------------

    dense_query = list(
        dense_model.embed([user_query])
    )[0].tolist()

    # ------------------------------------------------------
    # Generate sparse BM25 query embedding
    # ------------------------------------------------------

    sparse_query_raw = list(
        sparse_model.embed([user_query])
    )[0]

    sparse_query = models.SparseVector(
        indices=sparse_query_raw.indices.tolist(),
        values=sparse_query_raw.values.tolist(),
    )

    # ------------------------------------------------------
    # Filter to selected document
    # ------------------------------------------------------

    document_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(
                    value=file_id
                ),
            )
        ]
    )

    # ------------------------------------------------------
    # Execute hybrid search
    # ------------------------------------------------------

    search_results = qdrant_client.query_points(

        collection_name=COLLECTION_NAME,

        prefetch=[

            # ----------------------------------------------
            # Dense semantic search
            # ----------------------------------------------

            models.Prefetch(
                query=dense_query,
                using=VECTOR_NAME_DENSE,
                filter=document_filter,
                limit=candidate_k,
            ),

            # ----------------------------------------------
            # Sparse BM25 keyword search
            # ----------------------------------------------

            models.Prefetch(
                query=sparse_query,
                using=VECTOR_NAME_SPARSE,
                filter=document_filter,
                limit=candidate_k,
            ),
        ],

        # ----------------------------------------------
        # Reciprocal Rank Fusion
        # ----------------------------------------------

        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),

        # ----------------------------------------------
        # Number of final RRF candidates
        # ----------------------------------------------

        limit=candidate_k,

        # ----------------------------------------------
        # Payload contains chunk metadata/text
        # ----------------------------------------------

        with_payload=True,

        # ----------------------------------------------
        # We do not need vectors
        # ----------------------------------------------

        with_vectors=False,
    )

    return search_results.points


# ==========================================================
# STANDALONE TEST
# ==========================================================

def main() -> None:
    """
    Manual test for hybrid Qdrant retrieval.
    """

    user_query = (
        "What controls are used to protect applications, "
        "infrastructure, identities, and data?"
    )

    file_id = "OVERSIZED_PARENT_TEST_001"

    candidate_k = 10

    print()
    print("=" * 70)
    print("QDRANT HYBRID SEARCH TEST")
    print("=" * 70)

    print(f"Query       : {user_query}")
    print(f"File ID     : {file_id}")
    print(f"Candidate K : {candidate_k}")

    print()
    print("Initializing retriever...")

    initialize_retriever()

    print()
    print("Running hybrid search...")

    results = hybrid_search_rrf(
        user_query=user_query,
        file_id=file_id,
        candidate_k=candidate_k,
    )

    print()
    print("=" * 70)
    print(f"RESULTS: {len(results)} chunks")
    print("=" * 70)

    if not results:

        print(
            "\nNo matching chunks found for "
            f"file_id='{file_id}'."
        )

        return

    for rank, point in enumerate(
        results,
        start=1,
    ):

        payload = point.payload or {}

        print()
        print("-" * 70)

        print(f"RRF Rank       : {rank}")
        print(f"RRF Score      : {point.score:.6f}")
        print(f"Document ID    : {payload.get('document_id')}")
        print(f"Document Name  : {payload.get('document_name')}")
        print(f"Chunk ID       : {payload.get('chunk_id')}")
        print(f"Page           : {payload.get('page_no')}")
        print(f"Section Path   : {payload.get('section_path')}")
        print(f"Table Ref      : {payload.get('table_ref')}")

        print()
        print("Chunk Text:")
        print(payload.get("chunk_text", ""))

    print()
    print("=" * 70)
    print("HYBRID SEARCH TEST COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
