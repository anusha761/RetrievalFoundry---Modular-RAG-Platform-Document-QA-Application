"""
RERANKER - Cross-Encoder Reranking

Purpose
-------
Take candidate chunks produced by the Qdrant hybrid-search/RRF stage,
score each candidate against the user's query using a cross-encoder,
and return the top desired_k chunks.

Initialization
--------------
The cross-encoder is initialized explicitly through
initialize_reranker().

This prevents model loading on every API request.
"""

from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder


# ==========================================================
# CONFIGURATION
# ==========================================================

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


# ==========================================================
# RUNTIME RESOURCE
# ==========================================================

reranker: Optional[CrossEncoder] = None


# ==========================================================
# INITIALIZATION
# ==========================================================

def initialize_reranker() -> None:
    """
    Load the cross-encoder model once during application startup.

    Repeated calls are ignored if the model is already loaded.
    """

    global reranker

    if reranker is not None:
        return

    print("Loading cross-encoder reranker...")

    reranker = CrossEncoder(
        RERANKER_MODEL_NAME
    )

    print("Cross-encoder reranker loaded.")


# ==========================================================
# RESOURCE VALIDATION
# ==========================================================

def _ensure_initialized() -> None:
    """
    Ensure the reranker model has been initialized.
    """

    if reranker is None:

        raise RuntimeError(
            "Reranker is not initialized. "
            "Call initialize_reranker() during application startup."
        )


# ==========================================================
# RERANK FUNCTION
# ==========================================================

def rerank_chunks(
    user_query: str,
    candidate_chunks: List[Dict[str, Any]],
    desired_k: int,
) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks using a cross-encoder.

    Each returned chunk contains all original metadata plus:

        reranker_score
    """

    # ------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------

    if not isinstance(user_query, str):
        raise TypeError(
            "user_query must be a string."
        )

    user_query = user_query.strip()

    if not user_query:
        raise ValueError(
            "user_query cannot be empty."
        )

    if not isinstance(candidate_chunks, list):
        raise TypeError(
            "candidate_chunks must be a list."
        )

    if not isinstance(desired_k, int):
        raise TypeError(
            "desired_k must be an integer."
        )

    if desired_k <= 0:
        raise ValueError(
            "desired_k must be greater than 0."
        )

    if not candidate_chunks:
        return []

    # ------------------------------------------------------
    # Ensure model exists
    # ------------------------------------------------------

    _ensure_initialized()

    # ------------------------------------------------------
    # Limit desired_k
    # ------------------------------------------------------

    desired_k = min(
        desired_k,
        len(candidate_chunks),
    )

    # ------------------------------------------------------
    # Build query-document pairs
    # ------------------------------------------------------

    pairs = []

    for index, chunk in enumerate(
        candidate_chunks
    ):

        if not isinstance(chunk, dict):

            raise TypeError(
                f"Candidate chunk at index {index} "
                "must be a dictionary."
            )

        chunk_text = chunk.get(
            "chunk_text"
        )

        if not isinstance(chunk_text, str):

            raise ValueError(
                f"Candidate chunk at index {index} "
                "does not contain a valid "
                "'chunk_text' string."
            )

        chunk_text = chunk_text.strip()

        if not chunk_text:

            raise ValueError(
                f"Candidate chunk at index {index} "
                "has empty 'chunk_text'."
            )

        pairs.append(
            [
                user_query,
                chunk_text,
            ]
        )

    # ------------------------------------------------------
    # Generate cross-encoder scores
    # ------------------------------------------------------

    scores = reranker.predict(
        pairs
    )

    # ------------------------------------------------------
    # Attach scores
    # ------------------------------------------------------

    scored_chunks = []

    for chunk, score in zip(
        candidate_chunks,
        scores,
    ):

        reranked_chunk = dict(chunk)

        reranked_chunk["reranker_score"] = (
            float(score)
        )

        scored_chunks.append(
            reranked_chunk
        )

    # ------------------------------------------------------
    # Sort
    # ------------------------------------------------------

    scored_chunks.sort(
        key=lambda item: item[
            "reranker_score"
        ],
        reverse=True,
    )

    # ------------------------------------------------------
    # Return top K
    # ------------------------------------------------------

    return scored_chunks[
        :desired_k
    ]


# ==========================================================
# STANDALONE TEST
# ==========================================================

def main() -> None:
    """
    Standalone reranker test.

    Uses sample candidate chunks.
    """

    user_query = (
        "What controls are used to protect applications, "
        "infrastructure, identities, and data?"
    )

    desired_k = 5

    candidate_chunks = [
        {
            "document_name":
                "OVERSIZED_PARENT_TEST_001.pdf",

            "document_id":
                "OVERSIZED_PARENT_TEST_001",

            "chunk_id":
                "OVERSIZED_PARENT_TEST_001_chunk_0012",

            "page_no": 10,

            "section_path": (
                "Enterprise Technology Governance Report > "
                "Governance Framework > Security Governance"
            ),

            "table_ref": [],

            "chunk_text": (
                "### Security Governance\n\n"
                "Security governance establishes requirements "
                "for protecting applications, infrastructure, "
                "identities, data, and integrations.\n\n"
                "Security controls include identity management, "
                "access management, encryption, vulnerability "
                "management, logging, monitoring, incident "
                "response, and security testing."
            ),

            "retrieval_text": (
                "Enterprise Technology Governance Report > "
                "Governance Framework > Security Governance"
            ),
        },

        {
            "document_name":
                "OVERSIZED_PARENT_TEST_001.pdf",

            "document_id":
                "OVERSIZED_PARENT_TEST_001",

            "chunk_id":
                "OVERSIZED_PARENT_TEST_001_chunk_0013",

            "page_no": 11,

            "section_path": (
                "Enterprise Technology Governance Report > "
                "Governance Framework > Security Governance > "
                "Identity and Access Management"
            ),

            "table_ref": [],

            "chunk_text": (
                "#### Identity and Access Management\n\n"
                "Identity and access management controls "
                "determine who or what can access enterprise "
                "systems.\n\n"
                "Access should be granted according to business "
                "need and should follow the principle of "
                "least privilege."
            ),

            "retrieval_text": (
                "Identity and access management controls "
                "determine who or what can access enterprise "
                "systems."
            ),
        },
    ]

    print()
    print("=" * 70)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 70)

    print()
    print("Initializing reranker...")

    initialize_reranker()

    print()
    print("Running cross-encoder reranking...")

    results = rerank_chunks(
        user_query=user_query,
        candidate_chunks=candidate_chunks,
        desired_k=desired_k,
    )

    print()
    print("=" * 70)
    print(f"RESULTS: {len(results)} chunks")
    print("=" * 70)

    for rank, chunk in enumerate(
        results,
        start=1,
    ):

        print()
        print("-" * 70)

        print(
            f"Reranker Rank : {rank}"
        )

        print(
            f"Reranker Score: "
            f"{chunk['reranker_score']:.6f}"
        )

        print(
            f"Chunk ID      : "
            f"{chunk['chunk_id']}"
        )

        print(
            f"Page          : "
            f"{chunk['page_no']}"
        )

        print(
            f"Section Path  : "
            f"{chunk['section_path']}"
        )

        print()
        print("Chunk Text:")
        print(chunk["chunk_text"])


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()