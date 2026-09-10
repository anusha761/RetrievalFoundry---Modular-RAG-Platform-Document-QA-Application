"""Summary service: batch + final-summary generation.

Chunk-id handling:
  BATCH stage  -> LOCAL per-batch numbering: real chunk_id -> local number (1..N)
                  for the batch, wrap content with [chunk_id : <num>][file_name : X]
                  markers, call batch LLM, then map the numeric [chunk_id : N] in
                  ANSWER/PAGE_CONTENT BACK to the REAL chunk_id.
  FINAL stage  -> GLOBAL renumbering: real chunk_id -> global number, re-map the
                  concatenated batch ANSWERs, collapse consecutive duplicate tags,
                  transform_to_cid ([chunk_id : N] -> [c_id: N]), then call the
                  final-summary LLM. Citations are extracted from the final
                  response as [n : X][c_id : Y] pairs and resolved by GLOBAL number.
"""
import json
import re
import structlog
from datetime import datetime, timezone
from typing import List, Dict, Tuple
from app.config import load_prompt, load_settings
from app.interfaces.llm_interface import LLMInterface
from app.utils import fdi_helpers

logger = structlog.get_logger()


def _get_current_date() -> str:
    """Get current date as YYYY-MM-DD string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _format_conversation_history(conversation_history: List[dict]) -> List[dict]:
    """Convert stored history turns into chat-style message dicts for the /chat API.

    Each stored turn is {"prompt_question", "summary", "file_ids"}. It becomes a
    user message (the question plus the file ids it was asked over) followed by an
    assistant message (the answer):
        {"role": "user", "content": "<prompt_question> Files: <id1>, <id2>"}
        {"role": "assistant", "content": "<summary>"}

    The incoming list is most-recent-first; we REVERSE it so messages read
    oldest -> newest, which is the natural chat order.
    """
    messages: List[dict] = []
    for turn in reversed(conversation_history or []):
        question = turn.get("prompt_question", "") or ""
        file_ids = turn.get("file_ids", []) or []
        files_str = ", ".join(file_ids)
        user_content = f"{question} Files: {files_str}" if files_str else question
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": turn.get("summary", "") or ""})
    return messages


# ==============================================================
# JSON PARSING 
# ==============================================================

def _escape_control_chars_in_strings(text: str) -> str:
    """Escape raw control characters inside JSON string literals ."""
    def escape_match(match):
        content = match.group(0)
        return (content
                .replace('\n', '\\n')
                .replace('\r', '\\r')
                .replace('\t', '\\t'))
    return re.sub(r'"(.*?)"', escape_match, text, flags=re.DOTALL)


def _parse_llm_json(response_text: str) -> dict:
    """Parse an LLM JSON object response with progressive cleanup."""
    text = response_text.strip()

    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        cleaned = _escape_control_chars_in_strings(text)
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(f"summary_service.json_parse_failed - raw: {response_text}")
        return {"ANSWER": response_text, "PAGE_CONTENT": "", "SOURCE": ""}


def _parse_batch_response(chat_resp: str) -> List[dict]:
    """Parse a batch LLM response into a list of {PAGE_CONTENT, SOURCE, ANSWER} dicts.

    Slice to braces, wrap into JSON array, escape control chars inside strings, drop trailing commas.
    Returns a list.
    """
    text = chat_resp.strip()

    # Strip code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    if not text.startswith('{'):
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
        else:
            logger.info("summary_service.batch_parse - missing braces")

    data_str = "[" + text.strip() + "]"

    def escape_control_chars(match):
        content = match.group(0)
        return content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

    data_str = re.sub(r'"(.*?)"', escape_control_chars, data_str, flags=re.DOTALL)
    data_str = re.sub(r'\}\s*\{', r'},{', data_str)
    data_str = re.sub(r',\s*]', ']', data_str)
    data_str = re.sub(r',\s*}', '}', data_str)


    try:
        parsed = json.loads(data_str)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed
    except json.JSONDecodeError:
        logger.warning(f"summary_service.batch_parse_failed - raw: {chat_resp}")
        # return [{"ANSWER": chat_resp, "PAGE_CONTENT": "", "SOURCE": ""}]
        return []


# ==============================================================
# SINGLE-PASS NUMERIC TAGGING (route-0 style helpers)
# ==============================================================

def build_chunk_id_mapping(retrieved_chunks: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build bidirectional real chunk_id <-> sequential number maps (single-pass)."""
    real_to_num: Dict[str, int] = {}
    counter = 1
    for file_data in retrieved_chunks:
        for chunk in file_data["chunks"]:
            cid = chunk["chunk_id"]
            if cid not in real_to_num:
                real_to_num[cid] = counter
                counter += 1
    num_to_real = {v: k for k, v in real_to_num.items()}
    return real_to_num, num_to_real


def tag_chunks_with_ids(retrieved_chunks: List[Dict], real_to_num: Dict[str, int]) -> List[Dict]:
    """Tag each chunk's page_content with [chunk_id : N][file_name : X] markers."""
    tagged = []
    for file_data in retrieved_chunks:
        file_name = file_data["file_name"]
        tagged_file = {"file_id": file_data["file_id"], "file_name": file_name, "chunks": []}
        for chunk in file_data["chunks"]:
            num_id = real_to_num[chunk["chunk_id"]]
            section_path = chunk.get("section_path", [])
            section_headers = "\n".join(section_path) + "\n" if section_path else ""
            tagged_content = (
                f"[chunk_id : {num_id}][file_name : {file_name}]\n"
                f"{section_headers}"
                f"{chunk['page_content']} "
                f"[chunk_id : {num_id}][file_name : {file_name}]"
            )
            tagged_file["chunks"].append({
                "page_content": tagged_content,
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
                "section_path": section_path,
            })
        tagged.append(tagged_file)
    return tagged


def map_chunk_ids_back(text: str, num_to_real: Dict[int, str]) -> str:
    """Replace numeric [chunk_id : N] with real chunk_ids (single-pass)."""
    def replace_match(match):
        num_str = match.group(1).strip()
        try:
            num = int(num_str)
            return f"[chunk_id : {num_to_real.get(num, num_str)}]"
        except ValueError:
            return match.group(0)
    return re.sub(r'\[chunk_id\s*:\s*(\d+)\]', replace_match, text)


def strip_citation_markers(text: str) -> str:
    """Remove inline [chunk_id : X] / [file_name : X] / [c_id: X] markers."""
    if not text:
        return text
    text = re.sub(r'\[chunk_id\s*:\s*[^\]]+\]', '', text)
    text = re.sub(r'\[c_id\s*:\s*[^\]]+\]', '', text)
    text = re.sub(r'\[n\s*:\s*[^\]]+\]', '', text)
    text = re.sub(r'\[file_name\s*:\s*[^\]]+\]', '', text)
    text = re.sub(r'[ \t]+([,.;:)])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.strip()


# ==============================================================
# BATCH SUMMARY 
# ==============================================================

async def generate_batch_summary(
    llm: LLMInterface,
    question: str,
    batch_files: List[Dict],
    domain_knowledge: str,
    context: str = "",
) -> Tuple[List[dict], dict]:
    """Generate the per-batch summary using LOCAL numbering scheme.

    Args:
        llm: LLM service.
        question: user's (modified) question.
        batch_files: list of file dicts for THIS batch, each with UNtagged chunks
                     carrying REAL chunk_ids:
                     {"file_id","file_name","chunks":[{page_content,page_number,chunk_id,section_path}]}
        domain_knowledge: domain context string.
        context: optional extra context.

    Returns:
        Tuple of (batch_summary_list, token_usage). Each item in batch_summary_list
        is a dict with ANSWER / PAGE_CONTENT / SOURCE whose inline [chunk_id : ...]
        markers have been mapped back to REAL chunk_ids.
    """
    system_prompt = load_prompt("batch_summary_system")
    user_prompt_template = load_prompt("batch_summary_user")
    current_date = _get_current_date()
    settings = load_settings()
    json_response_format = getattr(settings, "MULTI_DOC_RESPONSE_FORMAT", "")

    # Flatten this batch's retriever response (real chunk_ids)
    retriever_resp: List[dict] = []
    batch_file_names: List[str] = []
    for file_data in batch_files:
        batch_file_names.append(file_data["file_name"])
        for chunk in file_data["chunks"]:
            retriever_resp.append({
                "page_content": chunk["page_content"],
                "source": file_data["file_name"],
                "page_number": chunk["page_number"],
                "chunk_id": chunk["chunk_id"],
            })

    # --- LOCAL per-batch numbering ---
    chunk_id_mapping = fdi_helpers.fetch_unique_chunk_ids(retriever_resp)
    retriever_resp = fdi_helpers.utilize_chunk_id_mapping(chunk_id_mapping, retriever_resp)
    chunk_num_to_id = {item["chunk_num"]: item["chunk_id"] for item in chunk_id_mapping}

    # Wrap each chunk with [chunk_id : <num>][file_name : <source>] markers
    for i in retriever_resp:
        i["page_content"] = (
            "[chunk_id : " + str(i["chunk_id"]) + "]" + "[file_name : " + str(i["source"]) + "] "
            + i["page_content"] + " [chunk_id : " + str(i["chunk_id"]) + "]" + "[file_name : " + str(i["source"]) + "]"
        )

    # Merge per source
    from collections import defaultdict
    merged_data = defaultdict(list)
    for item in retriever_resp:
        merged_data[item["source"]].append(item["page_content"])
    combined_resp = [{"page_content": " +++ ".join(content), "source": source}
                     for source, content in merged_data.items()]

    # Build prompts using escaped literal braces from the Python prompt modules.
    
    user_prompt = user_prompt_template.format(
        current_date=current_date,
        question=question,
        context=context or "",
        relevant_domain_knowledge=domain_knowledge or "",
        batch_size=len(batch_files),
        batch_files=", ".join(batch_file_names),
        json_response_format=json_response_format,
        relevant_content=json.dumps(combined_resp, indent=2),
    )

    logger.info(f"summary_service.batch_summary - batch_files: {batch_file_names} relevant_content: {(combined_resp)}")

    

    chat_resp, token_usage = await llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)

    logger.info(f"summary_service.batch_summary_complete - batch_files: {batch_file_names} batch response: {chat_resp}")

    parsed_list = _parse_batch_response(chat_resp)

    for obj in parsed_list:
        if isinstance(obj, dict):
            logger.info(f"summary_service.batch_parsed - ANSWER: {obj.get('ANSWER', '')} , SOURCE: {obj.get('SOURCE', '')}")

    

    # ---  map LOCAL numbers BACK to REAL chunk_ids in ANSWER + PAGE_CONTENT ---
    def replace_chunk_ids(match):
        chunk_num = match.group(1).strip()
        return f"[chunk_id : {chunk_num_to_id.get(chunk_num, chunk_num)}]"

    for obj in parsed_list:
        if "PAGE_CONTENT" in obj and isinstance(obj["PAGE_CONTENT"], str):
            obj["PAGE_CONTENT"] = re.sub(r'\[chunk_id\s*:\s*(\d+)\]', replace_chunk_ids, obj["PAGE_CONTENT"])
        if "ANSWER" in obj and isinstance(obj["ANSWER"], str):
            obj["ANSWER"] = re.sub(r'\[chunk_id\s*:\s*(\d+)\]', replace_chunk_ids, obj["ANSWER"])

    # restore REAL chunk_ids on retriever_resp + strip markers, so batches can be
    # aggregated into retrieved_content_with_pages.
    for entry in retriever_resp:
        entry["chunk_id"] = chunk_num_to_id.get(entry["chunk_id"], entry["chunk_id"])
        entry["page_content"] = re.sub(r'\s*\[chunk_id\s*:\s*\d+\]', '', entry["page_content"])
        entry["page_content"] = re.sub(r"\[file_name\s*:\s*.*?\]", '', entry["page_content"])
        entry["page_content"] = entry["page_content"].strip()

    logger.info(f"summary_service.batch_complete - token_usage: {token_usage}")
    return parsed_list, retriever_resp, token_usage


# ==============================================================
# FINAL / OVERALL SUMMARY
# ==============================================================

async def generate_final_summary(
    llm: LLMInterface,
    question: str,
    batch_summaries: List[dict],
    aggregated_batch_chunks: List[dict],
    processed_file_names: List[str],
    conversation_history: List[dict] = None,
) -> Tuple[str, str, List[dict], dict]:
    """Generate the final merged summary using GLOBAL renumbering chain.

    Args:
        llm: LLM service.
        question: user's question.
        batch_summaries: flat list of batch result dicts (ANSWER holds REAL chunk_ids).
        retrieved_chunks: GLOBAL retrieved chunks grouped by file (REAL chunk_ids,
                          UNtagged): [{"file_id","file_name","chunks":[...]}].
        processed_file_names: names of ALL processed files.
        conversation_history: optional prior turns.

    Returns:
        Tuple of:
          raw_llm_response      : the final LLM response text (contains [n:X][c_id:Y]),
          summary_clean         : cleaned user-facing summary (markers processed),
          numbered_pool         : flat list of retrieved chunks with chunk_id replaced
                                  by the GLOBAL number (for create_sourcesCollection),
          token_usage           : dict.
    """
    system_prompt = load_prompt("final_summary_system")
    user_prompt_template = load_prompt("final_summary_user")
    current_date = _get_current_date()

    # 1. Concatenate all batch ANSWERs (REAL chunk_ids). DEVIATION: "\n\n".join
    summary_collection = []
    for batch in batch_summaries:
        ans = batch.get("ANSWER", "") if isinstance(batch, dict) else ""
        if ans:
            summary_collection.append(ans)
    summary_concat = "\n\n".join(summary_collection)

    # 2. GLOBAL pool = aggregation of the chunks each batch ACTUALLY processed
    #    Numbering THIS aligns the global
    #    numbers with the batch ANSWERs' chunk_ids, so every file's [c_id] resolves.
    #    (Old code numbered the upfront-retrieval pool -> misaligned -> file dropped
    #    / all citations collapsed onto one chunk.)
    global_pool: List[dict] = [dict(item) for item in aggregated_batch_chunks]

    # 3. GLOBAL numbering
    chunk_id_mapping = fdi_helpers.fetch_unique_chunk_ids(global_pool)
    chunk_id_to_num = {item["chunk_id"]: item["chunk_num"] for item in chunk_id_mapping}

    # 4. Number the retrieved pool + wrap with markers (for create_sourcesCollection)
    numbered_pool = fdi_helpers.utilize_chunk_id_mapping(chunk_id_mapping, global_pool)
    for i in numbered_pool:
        i["page_content"] = (
            "[chunk_id : " + str(i["chunk_id"]) + "]" + "[file_name : " + str(i["source"]) + "] "
            + i["page_content"] + " [chunk_id : " + str(i["chunk_id"]) + "]" + "[file_name : " + str(i["source"]) + "]"
        )

    # 5. Re-map the concatenated batch answers: REAL chunk_id -> GLOBAL number
    def replace_chunk_ids(match):
        chunk_id = match.group(1).strip()
        return f"[chunk_id : {chunk_id_to_num.get(chunk_id, chunk_id)}]"

    summary_concat = re.sub(r'\[chunk_id\s*:\s*([^\]\s]+)\]', replace_chunk_ids, summary_concat)

    # 6. Collapse consecutive duplicate tags, then transform [chunk_id: N] -> [c_id: N]
    summary_concat = fdi_helpers.collapse_consecutive_chunk_ids(summary_concat)
    summary_concat = fdi_helpers.transform_to_cid(summary_concat)

    # 7. Build final-summary prompt
    
    user_prompt = user_prompt_template.format(
        current_date=current_date,
        question=question,
        count_processed_files=len(processed_file_names),
        processed_files=", ".join(processed_file_names),
        content=summary_concat,
    )

    # Conversation history is sent as structured chat messages (role/content),
    # NOT prepended to the user prompt.
    conv_messages = _format_conversation_history(conversation_history)

    logger.info(f"summary_service.final_summary - file_count: {len(processed_file_names)}")

    logger.info(f"summary_service.final_input: {len(summary_collection)}")
    logger.info(f"summary_Service.final_content: {summary_concat}")

    raw_llm_response, token_usage = await llm.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        conversation_history=conv_messages,
    )
    raw_llm_response = raw_llm_response.replace("<br>", " ")

    logger.info(f"summary_service.final_llm_response: {raw_llm_response}")

    # 8. Clean summary for display 
    summary_clean = fdi_helpers.summary_json_generation(raw_llm_response)
    if summary_clean is None:
        # Fallback: treat as plain string answer
        summary_clean = raw_llm_response
    summary_clean = fdi_helpers.clean_markdown_table(summary_clean)

    logger.info(f"summary_service.final_complete - token_usage: {token_usage}")
    return raw_llm_response, summary_clean, numbered_pool, token_usage


# ==============================================================
# SINGLE-PASS SUMMARY (Route-0 style; unchanged behaviour)
# ==============================================================

async def generate_single_pass_summary(
    llm: LLMInterface,
    question: str,
    retrieved_chunks: List[Dict],
    domain_knowledge: str,
    conversation_history: List[dict] = None
) -> Tuple[dict, dict]:
    """Generate summary in a single LLM call (used when file_count <= threshold).

    Uses route-0 style prompts (single_pass_system/user) that return
    {ANSWER, PAGE_CONTENT, SOURCE} JSON. Chunks here are pre-tagged by the caller.
    """
    system_prompt = load_prompt("single_pass_system")
    user_prompt_template = load_prompt("single_pass_user")
    current_date = _get_current_date()

    combined_content = []
    file_names = []
    for file_data in retrieved_chunks:
        file_name = file_data["file_name"]
        if file_name not in file_names:
            file_names.append(file_name)
        for chunk in file_data["chunks"]:
            combined_content.append({
                "page_content": chunk["page_content"],
                "source": file_name
            })

    
    user_prompt = user_prompt_template.format(
        current_date=current_date,
        question=question,
        domain_knowledge=domain_knowledge,
        count_files=len(file_names),
        file_names=", ".join(file_names),
        retrieved_content=json.dumps(combined_content, indent=2),
    )

    # Conversation history is sent as structured chat messages (role/content),
    # NOT prepended to the user prompt.
    conv_messages = _format_conversation_history(conversation_history)

    logger.info(f"summary_service.single_pass - file_count: {len(retrieved_chunks)}")

    response_text, token_usage = await llm.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        conversation_history=conv_messages,
    )
    parsed = _parse_llm_json(response_text)

    logger.info(f"summary_service.single_pass_complete - token_usage: {token_usage}")
    return parsed, token_usage
