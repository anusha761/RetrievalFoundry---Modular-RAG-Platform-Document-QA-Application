import re
import json
import structlog

logger = structlog.get_logger()


# ----------------------------------------------------------------------
# fetch_unique_chunk_ids  
# ----------------------------------------------------------------------
def fetch_unique_chunk_ids(retriever_response):
    """Map each unique real chunk_id to a sequential number (as string, 1-based).

    Returns: [{'chunk_id': <real_id>, 'chunk_num': '<n>'}, ...]
    """
    try:
        unique_chunk_ids = []
        for entry in retriever_response:
            chunk_id = entry['chunk_id']
            if chunk_id not in unique_chunk_ids:
                unique_chunk_ids.append(chunk_id)

        chunk_mapping = [{'chunk_id': cid, 'chunk_num': str(idx + 1)}
                         for idx, cid in enumerate(unique_chunk_ids)]
        return chunk_mapping
    except Exception as e:
        logger.error(f"fetch_unique_chunk_ids error: {e}")
        return retriever_response


# ----------------------------------------------------------------------
# utilize_chunk_id_mapping 
# ----------------------------------------------------------------------
def utilize_chunk_id_mapping(chunk_mapping, retriever_response):
    """Replace each entry's real chunk_id with its mapped number (string)."""
    mapping_dict = {m['chunk_id']: str(m['chunk_num']) for m in chunk_mapping}
    updated_data = []
    for entry in retriever_response:
        entry_copy = entry.copy()
        entry_copy['chunk_id'] = mapping_dict[entry_copy['chunk_id']]
        updated_data.append(entry_copy)
    return updated_data


# ----------------------------------------------------------------------
# collapse_consecutive_chunk_ids 
# ----------------------------------------------------------------------
def collapse_consecutive_chunk_ids(text: str) -> str:
    """Keep only the LAST occurrence of each [chunk_id : N] tag; drop earlier dupes.

    For each numeric chunk tag, only the final occurrence
    (by position) is retained in the output; all earlier occurrences are removed.
    """
    TAG_RE = re.compile(r"\s*\[(chunk_id)\s*:\s*(\d+)\]")
    matches = list(TAG_RE.finditer(text))

    last_for_id = {}
    for idx, m in enumerate(matches):
        chunk_num = m.group(2)
        last_for_id[chunk_num] = idx

    out = []
    cursor = 0
    for idx, m in enumerate(matches):
        out.append(text[cursor:m.start()])
        chunk_num = m.group(2)
        if last_for_id[chunk_num] == idx:
            out.append(m.group(0))
        cursor = m.end()
    out.append(text[cursor:])
    return "".join(out)


# ----------------------------------------------------------------------
# transform_to_cid 
# ----------------------------------------------------------------------
def transform_to_cid(text: str) -> str:
    """Convert [chunk_id : N] -> [c_id: N] and double the newlines."""
    text = re.sub(r"\[chunk_id\s*:\s*(\d+)\]", r"[c_id: \1]", text)
    text = text.replace("\n", "\n\n")
    return text


# ----------------------------------------------------------------------
# extract_citation_chunkid_pairs 
# ----------------------------------------------------------------------
def extract_citation_chunkid_pairs(text: str):
    """Extract [n : X][c_id : Y] pairs from the final LLM response.

    Returns: [{'citation': <int>, 'chunk_id': '<Y as str>'}, ...]
    The chunk_id here is the GLOBAL NUMBER (string), matching the numbered pool.
    """
    pattern = r"\[n\s*:\s*(\d+)\]\s*\[c_id\s*:\s*(\d+)\]"
    return [{"citation": int(n), "chunk_id": cid} for n, cid in re.findall(pattern, text)]


# ----------------------------------------------------------------------
# clean_json 
# ----------------------------------------------------------------------
def clean_json(raw_text: str) -> str:
    cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', raw_text)
    cleaned = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', cleaned)
    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
    cleaned = re.sub(r"[\n\t\r\f\v]", "", cleaned)
    cleaned = re.sub(r"'''|```|json", "", cleaned)
    cleaned = re.sub(r"^.*?({)", r"\1", cleaned)
    return cleaned


# ----------------------------------------------------------------------
# process_final_answer_r1 
# ----------------------------------------------------------------------
def process_final_answer_r1(text: str) -> str:
    """Convert inline [n : X]... markers into 'citation : X' at line ends."""
    output_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        if '[n' in line:
            sentence_part, citation_part = line.split('[n', 1)
            sentence = sentence_part.strip()
            citation_num = citation_part.split(']')[0].replace(':', '').strip()
            output_lines.append(f"{sentence} citation : {citation_num}")
        else:
            output_lines.append(line)
    return "\n\n".join(output_lines)


# ----------------------------------------------------------------------
# summary_json_generation 
# ----------------------------------------------------------------------
# def summary_json_generation(llm_response) -> str:
#     """Extract ANSWER from the final LLM JSON and post-process citations."""
#     try:
#         llm_response = str(llm_response)
#         if llm_response.rfind('}') == -1:
#             llm_response += '}'
#         llm_response = clean_json(llm_response)
#         json_start = llm_response.find('{')
#         json_end = llm_response.rfind('}')
#         json_str = llm_response[json_start:json_end + 1]
#         summary_json = json.loads(json_str)
#         summary_answer = summary_json['ANSWER']
#         summary = process_final_answer_r1(summary_answer)
#         summary = re.sub(r'citation\s*:\s*(\d+)', r'citation:\1', summary)
#         return summary
#     except Exception as e:
#         logger.error(f"summary_json_generation error: {e}")
#         return None


def summary_json_generation(llm_response) -> str:
    """Return CLEAN human-readable prose: strip [n]/[c_id] markers, collapse
    consecutive duplicate lines. Sources panel is unaffected (citations are
    resolved separately from the raw response)."""
    try:
        llm_response = str(llm_response)
        if llm_response.rfind('}') == -1:
            llm_response += '}'
        llm_response = clean_json(llm_response)
        json_start = llm_response.find('{')
        json_end = llm_response.rfind('}')
        json_str = llm_response[json_start:json_end + 1]
        summary_json = json.loads(json_str)
        summary_answer = summary_json['ANSWER']

        text = re.sub(r'\[n\s*:\s*\d+\]', '', summary_answer)
        text = re.sub(r'\[c_id\s*:\s*\d+\]', '', text)
        text = re.sub(r'\[chunk_id\s*:\s*[^\]]*\]', '', text)
        text = re.sub(r'\[file_name\s*:\s*[^\]]*\]', '', text)

        cleaned_lines = []
        for raw_line in text.split('\n'):
            line = re.sub(r'[ \t]{2,}', ' ', raw_line).strip()
            line = re.sub(r'\s+([,.:;])', r'\1', line)
            if not line:
                continue
            if cleaned_lines and cleaned_lines[-1] == line:
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)
    except Exception as e:
        logger.error(f"summary_json_generation error: {e}")
        return None


# ----------------------------------------------------------------------
# clean_markdown_table 
# ----------------------------------------------------------------------
def clean_markdown_table(md_text) -> str:
    try:
        md_text = str(md_text)
        lines = md_text.split('\n')
        cleaned_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith('|'):
                if line.strip():
                    cleaned_lines.append(line)
            else:
                if line.strip() != '':
                    cleaned_lines.append('')
                    cleaned_lines.append(line)
                    cleaned_lines.append('')
        return '\n'.join(cleaned_lines)
    except Exception:
        return md_text


# ----------------------------------------------------------------------
# create_sourcesCollection 
# ----------------------------------------------------------------------
def create_sourcesCollection(retrieved_content_with_pages, citations, file_details):
    """Resolve citations against the numbered retrieved pool.

    Args:
        retrieved_content_with_pages: flat list of chunks where 'chunk_id' has
            been replaced by the GLOBAL NUMBER (string), and 'page_content' has
            been wrapped with markers. Each item has: source, page_number, chunk_id.
        citations: [{'citation': int, 'chunk_id': '<global number str>'}]
        file_details: [{'file_name','file_id'}, ...] for file_id lookup.

    Returns:
        List of merged source dicts:
        {source_document, file_id, source(excerpt), page_number, chunk_id, citation}
    """
    sources_op = []
    for i in retrieved_content_with_pages:
        doc_source = {}
        doc_source['source_document'] = i['source']
        matched = [item['file_id'] for item in file_details
                   if item['file_name'] == doc_source['source_document']]
        if not matched:
            continue
        doc_source['file_id'] = matched[0]
        if doc_source['file_id'] == -1:
            continue

        # Strip markers from the excerpt
        pc = i['page_content']
        pc = re.sub(r's*\[chunk_id\s*:\s*\d+]', '', pc)
        pc = re.sub(r"\[file_name\s*:\s*.*?\]", '', pc)
        pc = re.sub(r"\[chunk_id\s*:\s*.*?\]", '', pc)

        doc_source['source'] = pc.strip()
        doc_source['page_number'] = i['page_number']
        doc_source['chunk_id'] = i['chunk_id']
        sources_op.append(doc_source)

    # Lookup by chunk_id (the GLOBAL number)
    source_lookup = {}
    for src in sources_op:
        source_lookup[src["chunk_id"]] = src

    sources_collection = []
    for cit in citations:
        key = cit["chunk_id"]
        if key in source_lookup:
            merged = source_lookup[key].copy()
            merged.update({"citation": cit["citation"]})
            sources_collection.append(merged)

    return sources_collection
