"""Citation Service:
  citations = extract_citation_chunkid_pairs(final_llm_response)   # [n:X][c_id:Y] pairs
  sources   = create_sourcesCollection(numbered_pool, citations, file_details)

The citation 'chunk_id' is the GLOBAL NUMBER (string) assigned during the final
summary stage, and it is resolved against the numbered retrieved pool whose
chunk_id was replaced by that same global number.
"""
import structlog
from typing import List, Dict
from app.schemas.responses import Citation
from app.utils import fdi_helpers

logger = structlog.get_logger()


def build_citations_fdi(
    final_llm_response: str,
    numbered_pool: List[Dict],
    retrieved_chunks: List[Dict],
) -> List[Citation]:
    """Build citations.

    Args:
        final_llm_response: the raw final-summary LLM response text containing
            [n : X][c_id : Y] citation pairs.
        numbered_pool: flat list of retrieved chunks where 'chunk_id' is the GLOBAL
            NUMBER (string) and 'page_content' is marker-wrapped. Produced by
            summary_service.generate_final_summary. Each item also carries the
            real source/page_number.
        retrieved_chunks: GLOBAL retrieved chunks grouped by file (for file_id +
            section_path lookup).

    Returns:
        List of Citation objects (only the chunks the final answer referenced).
    """
    # file_details for create_sourcesCollection (file_name -> file_id)
    file_details = [{"file_name": f["file_name"], "file_id": f["file_id"]}
                    for f in retrieved_chunks]

    # section_path lookup keyed by (file_name, page_number) from the original pool
    section_lookup: Dict[tuple, list] = {}
    for f in retrieved_chunks:
        for ch in f["chunks"]:
            section_lookup[(f["file_name"], ch["page_number"])] = ch.get("section_path", [])

    # 1. Extract [n:X][c_id:Y] pairs from the FINAL response
    citations = fdi_helpers.extract_citation_chunkid_pairs(final_llm_response)

    # 2. Resolve against the numbered pool 
    sources_collection = fdi_helpers.create_sourcesCollection(
        numbered_pool, citations, file_details
    )

    # 3. Map source dicts -> Citation objects (dedup by (file, page))
    seen = set()
    result: List[Citation] = []
    for src in sources_collection:
        file_name = src["source_document"]
        page_number = src["page_number"]
        key = (file_name, page_number)
        if key in seen:
            continue
        seen.add(key)
        result.append(Citation(
            file_name=file_name,
            file_id=src["file_id"],
            page_number=page_number,
            excerpt=src.get("source", ""),
            section_path=section_lookup.get((file_name, page_number), []),
        ))

    logger.info(
        f"citation_service.built_fdi - citation_count: {len(result)}, "
        f"pairs_extracted: {len(citations)}"
    )
    return result


# ==============================================================
# SINGLE-PASS citation builder (Route-0 style: {ANSWER, CITATIONS})
# ==============================================================

def build_citations_single_pass(
    answer_with_real_ids: str,
    retrieved_chunks: List[Dict],
) -> List[Citation]:
    """Build citations for the single-pass (route-0 style) response.

    The single-pass answer carries inline [chunk_id : <REAL id>] markers (numbers
    already mapped back to real ids). Extract those and resolve each against the
    retrieved chunks. Only the chunks the answer referenced become citations.
    """
    import re

    # chunk_id -> metadata
    lookup: Dict[str, dict] = {}
    for f in retrieved_chunks:
        for ch in f["chunks"]:
            lookup[ch["chunk_id"]] = {
                "file_name": f["file_name"],
                "file_id": f["file_id"],
                "page_number": ch["page_number"],
                "page_content": ch["page_content"],
                "section_path": ch.get("section_path", []),
            }

    referenced = set(m.strip() for m in re.findall(
        r'\[chunk_id\s*:\s*([^\]]+)\]', answer_with_real_ids or ""))

    seen = set()
    result: List[Citation] = []
    for chunk_id in referenced:
        info = lookup.get(chunk_id)
        if not info:
            continue
        key = (info["file_name"], info["page_number"])
        if key in seen:
            continue
        seen.add(key)
        result.append(Citation(
            file_name=info["file_name"],
            file_id=info["file_id"],
            page_number=info["page_number"],
            excerpt=info["page_content"],
            section_path=info["section_path"],
        ))

    logger.info(f"citation_service.built_single_pass - citation_count: {len(result)}")
    return result
