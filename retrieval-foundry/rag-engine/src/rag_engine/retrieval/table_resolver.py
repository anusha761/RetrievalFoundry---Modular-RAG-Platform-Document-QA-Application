
"""
TABLE RESOLVER

Purpose
-------
Resolve table references attached to retrieved chunks.

A chunk may contain:

    "table_ref": []

or:

    "table_ref": [
        "tables/AAPL_10-K_Sample_table_001.md"
    ]

For every referenced table, this module:

1. Resolves the table filename against the configured table directory.
2. Reads the original table Markdown.
3. Extracts the table number from the filename.
4. Appends the original table content to the chunk.
5. Preserves the original chunk text separately.

This module does NOT:

    - Perform retrieval
    - Perform reranking
    - Call an LLM
    - Construct the final answer
    - Modify Qdrant

It only enriches already-retrieved chunks with their
referenced original tables.
"""

import re
from pathlib import Path
from typing import Any, Dict, List


# ==========================================================
# CONFIGURATION
# ==========================================================

# TABLES_DIRECTORY = Path(
#     r"D:\VECTORDBCREATION\document_processing_db_creation"
#     r"\output_stage_1_doc_ingestion\tables"
# )

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TABLES_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "output_stage_1_doc_ingestion"
    / "tables"
)

TABLE_REF_PREFIX = "tables"


# ==========================================================
# TABLE NUMBER EXTRACTION
# ==========================================================

def extract_table_number(
    table_reference: str,
) -> str:
    """
    Extract the table number from a table filename.

    Example
    -------
    tables/AAPL_10-K_Sample_table_001.md
        -> "1"

    tables/AAPL_10-K_Sample_table_002.md
        -> "2"

    The filename is expected to follow:

        *_table_<number>.md
    """

    if not isinstance(table_reference, str):
        raise TypeError(
            "table_reference must be a string."
        )

    filename = Path(table_reference).name

    match = re.search(
        r"_table_(\d+)\.md$",
        filename,
        flags=re.IGNORECASE,
    )

    if not match:
        raise ValueError(
            "Could not extract table number from "
            f"table reference: '{table_reference}'"
        )

    # Convert "001" -> "1"
    return str(int(match.group(1)))


# ==========================================================
# TABLE FILE RESOLUTION
# ==========================================================

def resolve_table_path(
    table_reference: str,
) -> Path:
    """
    Resolve a table reference to an actual table file.

    Example
    -------
    tables/AAPL_10-K_Sample_table_001.md

    becomes:

    D:\\...\\output_stage_1_doc_ingestion\\tables\\
        AAPL_10-K_Sample_table_001.md

    Only the filename is used. This prevents a table_ref
    from escaping the configured tables directory.
    """

    if not isinstance(table_reference, str):
        raise TypeError(
            "table_reference must be a string."
        )

    table_reference = table_reference.strip()

    if not table_reference:
        raise ValueError(
            "table_reference cannot be empty."
        )

    filename = Path(table_reference).name

    if not filename.lower().endswith(".md"):
        raise ValueError(
            "Table reference must point to a Markdown "
            f"file: '{table_reference}'"
        )

    table_path = TABLES_DIRECTORY / filename

    # ------------------------------------------------------
    # Safety check
    # ------------------------------------------------------

    tables_root = TABLES_DIRECTORY.resolve()
    resolved_path = table_path.resolve()

    try:
        resolved_path.relative_to(tables_root)
    except ValueError:
        raise ValueError(
            "Resolved table path is outside the configured "
            f"tables directory: '{table_reference}'"
        )

    return resolved_path


# ==========================================================
# READ TABLE FILE
# ==========================================================

def read_table_file(
    table_reference: str,
) -> str:
    """
    Read the original table Markdown file.

    Returns
    -------
    str
        Complete table Markdown content.
    """

    table_path = resolve_table_path(
        table_reference
    )

    if not table_path.exists():
        raise FileNotFoundError(
            "Referenced table file does not exist: "
            f"'{table_path}'"
        )

    if not table_path.is_file():
        raise ValueError(
            "Referenced table path is not a file: "
            f"'{table_path}'"
        )

    return table_path.read_text(
        encoding="utf-8"
    ).strip()


# ==========================================================
# REMOVE DUPLICATE TABLE TITLE
# ==========================================================

def remove_existing_table_title(
    table_content: str,
) -> str:
    """
    Remove the first Markdown heading when it is a
    '# Table N' heading.

    Example:

        # Table 1

        **Source document:** ...

    becomes:

        **Source document:** ...

    This allows the resolver to add its own consistent
    'Table 1' heading without producing:

        Table 1

        # Table 1
    """

    lines = table_content.splitlines()

    if not lines:
        return ""

    first_line = lines[0].strip()

    if re.match(
        r"^#\s+Table\s+\d+\s*$",
        first_line,
        flags=re.IGNORECASE,
    ):
        return "\n".join(lines[1:]).strip()

    return table_content.strip()


# ==========================================================
# BUILD TABLE BLOCK
# ==========================================================

def build_table_block(
    table_reference: str,
) -> str:
    """
    Read a table file and convert it into the table block
    that will be appended to a retrieved chunk.

    Example output:

        ### Table 1

        **Source document:** AAPL_10-K_Sample.pdf

        **Source PDF page:** 2

        **Docling reference:** `#/tables/0`

        ## Original Table

        | ... |
    """

    table_number = extract_table_number(
        table_reference
    )

    table_content = read_table_file(
        table_reference
    )

    table_content = remove_existing_table_title(
        table_content
    )

    return (
        f"### Table {table_number}\n\n"
        f"{table_content}"
    ).strip()


# ==========================================================
# RESOLVE TABLES FOR ONE CHUNK
# ==========================================================

def resolve_tables_for_chunk(
    chunk: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve all table references belonging to one chunk.

    Parameters
    ----------
    chunk:
        Retrieved chunk dictionary.

    Returns
    -------
    Dict[str, Any]
        Copy of the chunk enriched with table content.

    Additional fields
    -----------------
    original_chunk_text:
        Original chunk text before table resolution.

    resolved_table_refs:
        Successfully resolved table references.

    table_resolution_status:
        One of:

            "no_tables"
            "resolved"
            "partial"

    Notes
    -----
    The original chunk dictionary is not modified.
    """

    if not isinstance(chunk, dict):
        raise TypeError(
            "chunk must be a dictionary."
        )

    resolved_chunk = dict(chunk)

    original_chunk_text = chunk.get(
        "chunk_text",
        "",
    )

    if not isinstance(original_chunk_text, str):
        raise ValueError(
            "chunk must contain a valid 'chunk_text' string."
        )

    original_chunk_text = original_chunk_text.strip()

    resolved_chunk["original_chunk_text"] = (
        original_chunk_text
    )

    table_refs = chunk.get(
        "table_ref",
        [],
    )

    # ------------------------------------------------------
    # No table references
    # ------------------------------------------------------

    if not table_refs:

        resolved_chunk["resolved_table_refs"] = []

        resolved_chunk["table_resolution_status"] = (
            "no_tables"
        )

        return resolved_chunk

    if not isinstance(table_refs, list):
        raise ValueError(
            "'table_ref' must be a list."
        )

    table_blocks: List[str] = []
    resolved_table_refs: List[str] = []
    failed_table_refs: List[str] = []

    # ------------------------------------------------------
    # Resolve every referenced table
    # ------------------------------------------------------

    for table_reference in table_refs:

        if not isinstance(
            table_reference,
            str,
        ):
            raise ValueError(
                "Every value in 'table_ref' must be "
                "a string."
            )

        try:

            table_block = build_table_block(
                table_reference
            )

            table_blocks.append(
                table_block
            )

            resolved_table_refs.append(
                table_reference
            )

        except (
            FileNotFoundError,
            ValueError,
            OSError,
        ) as exc:

            failed_table_refs.append(
                table_reference
            )

            raise RuntimeError(
                "Failed to resolve table reference "
                f"'{table_reference}' for chunk "
                f"'{chunk.get('chunk_id')}'. "
                f"Reason: {exc}"
            ) from exc

    # ------------------------------------------------------
    # Build enriched chunk text
    # ------------------------------------------------------

    enriched_parts = []

    if original_chunk_text:
        enriched_parts.append(
            original_chunk_text
        )

    enriched_parts.append(
        "The original tables are mentioned below."
    )

    enriched_parts.extend(
        table_blocks
    )

    enriched_text = "\n\n".join(
        enriched_parts
    ).strip()

    # ------------------------------------------------------
    # Store enriched content
    # ------------------------------------------------------

    resolved_chunk["chunk_text"] = (
        enriched_text
    )

    resolved_chunk["resolved_table_refs"] = (
        resolved_table_refs
    )

    resolved_chunk["table_resolution_status"] = (
        "resolved"
    )

    return resolved_chunk


# ==========================================================
# RESOLVE TABLES FOR MULTIPLE CHUNKS
# ==========================================================

def resolve_tables_for_chunks(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Resolve table references for all final retrieved chunks.

    Parameters
    ----------
    chunks:
        Final reranked chunks.

    Returns
    -------
    List[Dict[str, Any]]
        Enriched chunks ready for the downstream LLM.
    """

    if not isinstance(chunks, list):
        raise TypeError(
            "chunks must be a list."
        )

    return [
        resolve_tables_for_chunk(chunk)
        for chunk in chunks
    ]


# ==========================================================
# STANDALONE TEST
# ==========================================================

def main() -> None:
    """
    Standalone test using the AAPL table reference supplied
    in the Stage 2 chunk metadata.
    """

    test_chunk = {
        "document_name": "AAPL_10-K_Sample.pdf",
        "document_id": "AAPL_10-K_Sample",
        "chunk_id": "AAPL_10-K_Sample_chunk_0008",
        "page_no": 2,
        "page_end": 2,
        "section_path": [
            "CONSOLIDATED STATEMENTS OF OPERATIONS",
            "Table 1 — Summary",
        ],
        "table_ref": [
            "tables/AAPL_10-K_Sample_table_001.md"
        ],
        "chunk_text": (
            "**Original table:** "
            "`tables/AAPL_10-K_Sample_table_001.md`"
        ),
        "token_count": 18,
    }

    print()
    print("=" * 70)
    print("TABLE RESOLVER TEST")
    print("=" * 70)

    print()
    print("Chunk ID:")
    print(test_chunk["chunk_id"])

    print()
    print("Table References:")
    print(test_chunk["table_ref"])

    print()
    print("Resolving table...")

    result = resolve_tables_for_chunk(
        test_chunk
    )

    print()
    print("=" * 70)
    print("RESOLVED CHUNK")
    print("=" * 70)

    print()
    print("Original Chunk Text:")
    print(result["original_chunk_text"])

    print()
    print("Resolved Table References:")
    print(result["resolved_table_refs"])

    print()
    print("Resolution Status:")
    print(result["table_resolution_status"])

    print()
    print("Final Chunk Text:")
    print("-" * 70)
    print(result["chunk_text"])

    print()
    print("=" * 70)
    print("TABLE RESOLVER TEST COMPLETED")
    print("=" * 70)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()

