"""
STAGE 2 - STRUCTURE-AWARE / HIERARCHICAL CHUNKING

Purpose
-------
Convert Stage 1 Markdown documents into retrieval-ready chunks while:

1. Preserving Markdown headings and hierarchy.
2. Preserving semantically useful heading-only sections.
3. Keeping logical sections together whenever possible.
4. Descending into child sections when a parent is too large.
5. Attaching structural heading-only headings to the next meaningful
   sibling instead of silently deleting them.
6. Preserving table references.
7. Preserving PDF page metadata.
8. Enforcing a hard maximum token limit.
9. Validating all chunks BEFORE writing them to chunks.jsonl.

IMPORTANT HEADING RULE
----------------------
A heading-only section is NOT automatically discarded.

Example:

    ## FORM 10-K

    ## Apple Inc.

    (Exact name of registrant...)

Both headings carry semantic information.

Because FORM 10-K and Apple Inc. are siblings, FORM 10-K is NOT placed
inside section_path for Apple Inc.

Instead, FORM 10-K is attached to the first meaningful chunk produced
for Apple Inc.

Result:

    ## FORM 10-K

    ## Apple Inc.

    (Exact name of registrant...)

This preserves the heading for retrieval without creating a
heading-only/orphan chunk.

INPUT
-----
Stage 1 Markdown files
Stage 1 manifest JSON files

OUTPUT
------
1. chunks.jsonl
2. stage_2_manifest.json
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tiktoken
except ImportError as exc:
    raise ImportError(
        "tiktoken is required for Stage 2.\n"
        "Install it with:\n"
        "    pip install tiktoken"
    ) from exc


# ============================================================
# CONFIGURATION
# ============================================================

# PROJECT_ROOT = Path(
#     r"D:/VECTORDBCREATION/document_processing_db_creation"
# )

# # Stage 1 outputs
# DOCUMENTS_DIR = (
#     PROJECT_ROOT
#     / "output_stage_1_doc_ingestion/documents"
# )

# MANIFESTS_DIR = (
#     PROJECT_ROOT
#     / "output_stage_1_doc_ingestion/manifests"
# )

# # Stage 2 outputs
# OUTPUT_DIR = (
#     PROJECT_ROOT
#     / "output_stage_2_chunk_metadata_generation"
# )

# CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"

# STAGE2_MANIFEST_FILE = (
#     OUTPUT_DIR / "stage_2_manifest.json"
# )

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Stage 1 outputs
STAGE1_DIR = PROJECT_ROOT / "data" / "stage_1"

DOCUMENTS_DIR = STAGE1_DIR / "documents"

MANIFESTS_DIR = STAGE1_DIR / "manifests"

# Stage 2 outputs
STAGE2_DIR = PROJECT_ROOT / "data" / "stage_2"

CHUNKS_FILE = STAGE2_DIR / "chunks.jsonl"

STAGE2_MANIFEST_FILE = STAGE2_DIR / "stage_2_manifest.json"


# ============================================================
# CHUNK CONFIGURATION
# ============================================================

TOKEN_COUNT_MODEL = "gpt-4o-mini"

# Keep your requested experimental value.
MAX_CHUNK_TOKENS = 7000

# Used only when token-window splitting becomes necessary.
OVERLAP_TOKENS = 20


# ============================================================
# TOKENIZER
# ============================================================

try:
    ENCODING = tiktoken.encoding_for_model(
        TOKEN_COUNT_MODEL
    )
except KeyError as exc:
    raise RuntimeError(
        f"Your installed tiktoken version does not recognize "
        f"{TOKEN_COUNT_MODEL!r}.\n"
        f"Upgrade tiktoken with:\n"
        f"    pip install --upgrade tiktoken"
    ) from exc


def count_tokens(text: str) -> int:
    return len(
        ENCODING.encode(text)
    )


def encode_tokens(text: str) -> List[int]:
    return ENCODING.encode(text)


def decode_tokens(tokens: List[int]) -> str:
    return ENCODING.decode(tokens)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class MarkdownUnit:
    unit_type: str
    text: str

    heading_level: Optional[int] = None
    heading_text: Optional[str] = None

    table_ref: Optional[str] = None

    page_no: Optional[int] = None
    page_end: Optional[int] = None

    section_path: List[str] = field(
        default_factory=list
    )


@dataclass
class SectionNode:
    heading: Optional[MarkdownUnit] = None

    content: List[MarkdownUnit] = field(
        default_factory=list
    )

    children: List["SectionNode"] = field(
        default_factory=list
    )


@dataclass
class Chunk:
    text: str

    page_no: Optional[int]
    page_end: Optional[int]

    section_path: List[str]

    table_ref: List[str]

    token_count: int


# ============================================================
# MARKDOWN REGEX
# ============================================================

HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)

FENCE_RE = re.compile(
    r"^\s*(```|~~~)"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip().lower()


def normalize_table_ref(
    table_ref: str,
) -> str:

    ref = (
        table_ref
        .strip()
        .replace("\\", "/")
    )

    ref = ref.lstrip("./")

    return ref.lower()


# ============================================================
# MANIFEST DISCOVERY
# ============================================================

def discover_manifest(
    markdown_file: Path,
    manifests_dir: Path,
) -> Path:

    document_id = markdown_file.stem

    candidates = sorted(
        manifests_dir.rglob("*.json")
    )

    # --------------------------------------------------------
    # 1. Exact filename stem match.
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate.stem == document_id:
            return candidate

    # --------------------------------------------------------
    # 2. Search inside JSON.
    # --------------------------------------------------------

    for candidate in candidates:

        try:

            with candidate.open(
                "r",
                encoding="utf-8",
            ) as f:

                data = json.load(f)

        except (
            OSError,
            json.JSONDecodeError,
        ):
            continue

        if data.get("document_id") == document_id:
            return candidate

        document_name = data.get(
            "document_name"
        )

        if (
            document_name
            and Path(document_name).stem
            == document_id
        ):
            return candidate

    raise FileNotFoundError(
        f"Could not find Stage 1 manifest for:\n"
        f"Markdown: {markdown_file}\n"
        f"Expected document_id: {document_id}"
    )


# ============================================================
# PAGE MAPPER
# ============================================================

class ManifestPageMapper:
    """
    Maps Markdown units to source PDF pages using
    the Stage 1 manifest.

    IMPORTANT PAGE-CITATION RULE
    ----------------------------
    For ordinary content, the page number used for the chunk
    should represent the immediate logical section heading.

    Therefore section headings are resolved from BOTH:

        1. SectionHeaderItem
        2. TextItem

    This is necessary because Docling may represent a logical
    heading as either element type.
    """

    def __init__(
        self,
        manifest: dict,
    ):

        self.text_elements: List[dict] = []

        self.table_elements: Dict[
            str,
            dict
        ] = {}

        # ----------------------------------------------------
        # Heading page lookup.
        #
        # IMPORTANT:
        #
        # A logical heading may appear in the manifest as either
        # SectionHeaderItem OR TextItem.
        #
        # Therefore both are registered here.
        # ----------------------------------------------------

        self.heading_pages: Dict[
            str,
            int
        ] = {}

        # ----------------------------------------------------
        # Text elements
        # ----------------------------------------------------

        for element in manifest.get(
            "elements",
            [],
        ):

            element_type = element.get(
                "element_type"
            )

            page_start = element.get(
                "page_start"
            )

            page_end = (
                element.get("page_end")
                or page_start
            )

            # ------------------------------------------------
            # Tables
            # ------------------------------------------------

            if element_type == "TableItem":

                table_ref = (
                    element.get("table_ref")
                    or element.get("table_file")
                )

                if (
                    table_ref
                    and page_start is not None
                ):

                    self._register_table_ref(
                        table_ref,
                        page_start,
                        page_end,
                    )

                continue

            # ------------------------------------------------
            # Normal text
            # ------------------------------------------------

            text = element.get(
                "text",
                "",
            )

            if (
                text
                and page_start is not None
            ):

                normalized_text = normalize_text(
                    text
                )

                self.text_elements.append(
                    {
                        "text":
                            normalized_text,

                        "page_start":
                            page_start,

                        "page_end":
                            page_end,
                    }
                )

                # ------------------------------------------------
                # IMPORTANT:
                #
                # A heading can be represented by either:
                #
                #   SectionHeaderItem
                #   TextItem
                #
                # We therefore register BOTH as heading
                # candidates.
                #
                # The lookup is exact against section_path heading
                # text, so arbitrary surrounding text does not
                # become part of the section path.
                # ------------------------------------------------

                if element_type in (
                    "SectionHeaderItem",
                    "TextItem",
                ):

                    self.heading_pages[
                        normalized_text
                    ] = page_start

        # ----------------------------------------------------
        # Top-level Stage 1 tables
        # ----------------------------------------------------

        for table in manifest.get(
            "tables",
            [],
        ):

            table_file = table.get(
                "table_file"
            )

            page_start = table.get(
                "page_start"
            )

            page_end = (
                table.get("page_end")
                or page_start
            )

            if (
                table_file
                and page_start is not None
            ):

                self._register_table_ref(
                    table_file,
                    page_start,
                    page_end,
                )

            # Alternate mappings
            for alias_key in (
                "table_ref",
                "docling_ref",
            ):

                alias = table.get(
                    alias_key
                )

                if (
                    alias
                    and page_start is not None
                ):

                    self._register_table_ref(
                        alias,
                        page_start,
                        page_end,
                    )

    def _register_table_ref(
        self,
        table_ref: str,
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> None:

        key = normalize_table_ref(
            table_ref
        )

        self.table_elements[key] = {
            "page_start": page_start,
            "page_end": page_end,
        }

    def find_pages_for_text(
        self,
        text: str,
    ) -> Tuple[
        Optional[int],
        Optional[int],
    ]:

        normalized = normalize_text(text)

        if not normalized:
            return None, None

        # Exact match
        for element in self.text_elements:

            if (
                element["text"]
                == normalized
            ):

                return (
                    element["page_start"],
                    element["page_end"],
                )

        return None, None

    # --------------------------------------------------------
    # HEADING PAGE LOOKUP
    # --------------------------------------------------------

    def find_heading_page(
        self,
        heading_text: str,
    ) -> Optional[int]:

        """
        Return the page_start of a logical heading found in
        the Stage 1 manifest.

        The heading may have been represented by Docling as
        either a SectionHeaderItem or a TextItem.
        """

        normalized = normalize_text(
            heading_text
        )

        if not normalized:
            return None

        return self.heading_pages.get(
            normalized
        )

    def find_fallback_page_from_section_path(
        self,
        section_path: List[str],
    ) -> Optional[int]:

        """
        Return the page of the immediate section heading.

        The section_path is ordered from outermost to innermost
        heading, so the last element is the immediate logical
        section heading for the content.

        We first try the immediate heading. If that heading
        cannot be resolved, walk outward as a defensive fallback.
        """

        if not section_path:
            return None

        # Immediate section heading has priority.
        immediate_heading = section_path[-1]

        page = self.find_heading_page(
            immediate_heading
        )

        if page is not None:
            return page

        # Defensive fallback to parent headings.
        for heading in reversed(
            section_path[:-1]
        ):

            page = self.find_heading_page(
                heading
            )

            if page is not None:
                return page

        return None

    def find_pages_for_table(
        self,
        table_ref: str,
    ) -> Tuple[
        Optional[int],
        Optional[int],
    ]:

        key = normalize_table_ref(
            table_ref
        )

        info = self.table_elements.get(
            key
        )

        if not info:
            return None, None

        return (
            info["page_start"],
            info["page_end"],
        )


# ============================================================
# TABLE REFERENCE
# ============================================================

def extract_table_ref(
    text: str,
) -> Optional[str]:

    patterns = (
        r"\*\*Original table:\*\*\s*`([^`]+)`",
        r"Original table:\s*`([^`]+)`",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1).strip()

    return None


# ============================================================
# MARKDOWN BLOCK SPLITTING
# ============================================================

def split_markdown_into_blocks(
    markdown: str,
) -> List[str]:

    lines = (
        markdown
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    )

    blocks: List[str] = []

    current: List[str] = []

    inside_fence = False
    fence_marker: Optional[str] = None

    def flush() -> None:

        nonlocal current

        if current:

            text = "\n".join(
                current
            ).strip()

            if text:
                blocks.append(text)

            current = []

    for line in lines:

        fence_match = FENCE_RE.match(
            line
        )

        # ----------------------------------------------------
        # Fenced code
        # ----------------------------------------------------

        if fence_match:

            marker = fence_match.group(1)

            if not inside_fence:

                flush()

                inside_fence = True
                fence_marker = marker

                current.append(line)

            else:

                current.append(line)

                if marker == fence_marker:

                    inside_fence = False
                    fence_marker = None

                    flush()

            continue

        if inside_fence:

            current.append(line)

            continue

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if HEADING_RE.match(line):

            flush()

            blocks.append(
                line.strip()
            )

            continue

        # ----------------------------------------------------
        # Blank line
        # ----------------------------------------------------

        if not line.strip():

            flush()

            continue

        current.append(line)

    flush()

    return blocks


# ============================================================
# MARKDOWN CLASSIFICATION
# ============================================================

def is_heading(
    text: str,
) -> bool:

    return bool(
        HEADING_RE.match(
            text.strip()
        )
    )


def parse_heading(
    text: str,
) -> Tuple[int, str]:

    match = HEADING_RE.match(
        text.strip()
    )

    if not match:

        raise ValueError(
            f"Not a Markdown heading: {text}"
        )

    return (
        len(match.group(1)),
        match.group(2).strip(),
    )


def is_code_block(
    text: str,
) -> bool:

    lines = text.splitlines()

    return (
        len(lines) >= 2
        and bool(
            FENCE_RE.match(
                lines[0]
            )
        )
    )


def is_list_block(
    text: str,
) -> bool:

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return False

    list_lines = sum(
        bool(
            re.match(
                r"^([-*+]|\d+\.)\s+",
                line,
            )
        )
        for line in lines
    )

    return list_lines >= max(
        1,
        len(lines) // 2,
    )


# ============================================================
# SECTION TRACKER
# ============================================================

class SectionTracker:

    def __init__(self):

        self.headers: Dict[
            int,
            str
        ] = {}

    def update(
        self,
        level: int,
        heading: str,
    ) -> None:

        # Remove current level and deeper levels
        for existing_level in list(
            self.headers
        ):

            if existing_level >= level:

                del self.headers[
                    existing_level
                ]

        self.headers[level] = heading

    def get_path(self) -> List[str]:

        return [
            self.headers[level]
            for level in sorted(
                self.headers
            )
        ]


# ============================================================
# BUILD MARKDOWN UNITS
# ============================================================

def build_markdown_units(
    markdown: str,
    page_mapper: ManifestPageMapper,
) -> List[MarkdownUnit]:

    units: List[MarkdownUnit] = []

    tracker = SectionTracker()

    for block in split_markdown_into_blocks(
        markdown
    ):

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if is_heading(block):

            level, heading = parse_heading(
                block
            )

            tracker.update(
                level,
                heading,
            )

            # Heading itself is mapped directly.
            page_no, page_end = (
                page_mapper.find_pages_for_text(
                    heading
                )
            )

            # If direct mapping fails, use the heading lookup.
            if page_no is None:

                heading_page = (
                    page_mapper.find_heading_page(
                        heading
                    )
                )

                if heading_page is not None:

                    page_no = heading_page
                    page_end = heading_page

            units.append(
                MarkdownUnit(
                    unit_type="heading",
                    text=block,
                    heading_level=level,
                    heading_text=heading,
                    page_no=page_no,
                    page_end=page_end,
                    section_path=(
                        tracker.get_path()
                    ),
                )
            )

            continue

        # ----------------------------------------------------
        # Current logical section
        # ----------------------------------------------------

        section_path = (
            tracker.get_path()
        )

        # ----------------------------------------------------
        # IMMEDIATE SECTION PAGE
        #
        # This is the page used for citation purposes.
        #
        # Example:
        #
        # ## CONSOLIDATED STATEMENTS OF OPERATIONS
        #
        # page = 2
        #
        # Every paragraph/table-summary belonging directly to
        # this section receives page 2, even if the individual
        # text happens to be mapped elsewhere in the manifest.
        # ----------------------------------------------------

        section_page = (
            page_mapper
            .find_fallback_page_from_section_path(
                section_path
            )
        )

        # ----------------------------------------------------
        # Table summary
        # ----------------------------------------------------

        table_ref = extract_table_ref(
            block
        )

        if table_ref:

            table_page_no, table_page_end = (
                page_mapper.find_pages_for_table(
                    table_ref
                )
            )

            # ------------------------------------------------
            # IMPORTANT:
            #
            # For citation consistency, the immediate section
            # heading page takes priority.
            #
            # If the section heading cannot be resolved,
            # retain the table's own page metadata.
            # ------------------------------------------------

            if section_page is not None:

                page_no = section_page
                page_end = section_page

            else:

                page_no = table_page_no
                page_end = table_page_end

            units.append(
                MarkdownUnit(
                    unit_type="table_summary",
                    text=block,
                    table_ref=table_ref,
                    page_no=page_no,
                    page_end=page_end,
                    section_path=section_path,
                )
            )

            continue

        # ----------------------------------------------------
        # Other content
        # ----------------------------------------------------

        unit_type = "paragraph"

        if is_code_block(block):

            unit_type = "code_block"

        elif is_list_block(block):

            unit_type = "list"

        # ----------------------------------------------------
        # IMPORTANT PAGE RULE
        #
        # Do NOT use direct substring/text page matching as the
        # primary page number for ordinary content.
        #
        # The immediate logical section heading determines the
        # citation page.
        #
        # If no section heading can be resolved, only then fall
        # back to direct manifest text matching.
        # ----------------------------------------------------

        if section_page is not None:

            page_no = section_page
            page_end = section_page

        else:

            page_no, page_end = (
                page_mapper.find_pages_for_text(
                    block
                )
            )

        units.append(
            MarkdownUnit(
                unit_type=unit_type,
                text=block,
                page_no=page_no,
                page_end=page_end,
                section_path=section_path,
            )
        )

    return units


# ============================================================
# BUILD SECTION TREE
# ============================================================

def build_section_tree(
    units: List[MarkdownUnit],
) -> SectionNode:

    root = SectionNode()

    stack: List[SectionNode] = [
        root
    ]

    for unit in units:

        # ----------------------------------------------------
        # Normal content
        # ----------------------------------------------------

        if unit.unit_type != "heading":

            stack[-1].content.append(
                unit
            )

            continue

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        level = (
            unit.heading_level
            or 1
        )

        # Same-level headings are siblings.
        # A deeper heading becomes a child.
        while len(stack) > 1:

            current_heading = (
                stack[-1].heading
            )

            current_level = (
                current_heading.heading_level
                if current_heading is not None
                else 0
            )

            if current_level < level:
                break

            stack.pop()

        node = SectionNode(
            heading=unit
        )

        stack[-1].children.append(
            node
        )

        stack.append(node)

    return root


# ============================================================
# PAGE RANGE
# ============================================================

def combine_page_ranges(
    units: List[MarkdownUnit],
) -> Tuple[
    Optional[int],
    Optional[int],
]:

    pages: List[int] = []

    for unit in units:

        if unit.page_no is not None:
            pages.append(unit.page_no)

        if unit.page_end is not None:
            pages.append(unit.page_end)

    if not pages:
        return None, None

    return (
        min(pages),
        max(pages),
    )


# ============================================================
# CHUNK CREATION
# ============================================================

def make_chunk_from_units(
    units: List[MarkdownUnit],
    text_override: Optional[str] = None,
    section_path_override: Optional[List[str]] = None,
) -> Optional[Chunk]:

    if (
        not units
        and text_override is None
    ):
        return None

    if text_override is not None:

        text = text_override.strip()

    else:

        text = "\n\n".join(
            unit.text.strip()
            for unit in units
            if unit.text.strip()
        ).strip()

    if not text:
        return None

    # --------------------------------------------------------
    # section_path
    #
    # IMPORTANT:
    #
    # section_path identifies the logical section that owns
    # this chunk. It must NOT simply be taken from the last
    # Markdown unit in the chunk.
    #
    # Example:
    #
    # ## Financial Performance
    # ### Revenue
    # ### Expenses
    # ### Profitability
    #
    # If all of this fits in one chunk, the section_path is:
    #
    # ["Apple Inc. Test Document",
    #  "Financial Performance"]
    #
    # NOT:
    #
    # ["Apple Inc. Test Document",
    #  "Financial Performance",
    #  "Profitability"]
    # --------------------------------------------------------

    if section_path_override is not None:

        section_path = (
            section_path_override.copy()
        )

    else:

        # Prefer the first heading in the unit group.
        heading_units = [
            unit
            for unit in units
            if unit.unit_type == "heading"
        ]

        if heading_units:

            section_path = (
                heading_units[0]
                .section_path.copy()
            )

        else:

            section_path = (
                units[0].section_path.copy()
                if units
                else []
            )

    # --------------------------------------------------------
    # table references
    # --------------------------------------------------------

    table_refs: List[str] = []

    for unit in units:

        if (
            unit.table_ref
            and unit.table_ref
            not in table_refs
        ):

            table_refs.append(
                unit.table_ref
            )

    # --------------------------------------------------------
    # pages
    # --------------------------------------------------------

    page_no, page_end = (
        combine_page_ranges(
            units
        )
    )

    return Chunk(
        text=text,
        page_no=page_no,
        page_end=page_end,
        section_path=section_path,
        table_ref=table_refs,
        token_count=count_tokens(text),
    )


# ============================================================
# TEXT SPLITTING
# ============================================================

def split_by_token_window(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:

    tokens = encode_tokens(text)

    if len(tokens) <= max_tokens:

        return [
            text.strip()
        ]

    if overlap_tokens >= max_tokens:

        raise ValueError(
            "OVERLAP_TOKENS must be smaller "
            "than MAX_CHUNK_TOKENS."
        )

    pieces: List[str] = []

    start = 0

    while start < len(tokens):

        end = min(
            start + max_tokens,
            len(tokens),
        )

        piece = decode_tokens(
            tokens[start:end]
        ).strip()

        if piece:
            pieces.append(piece)

        if end >= len(tokens):
            break

        start = (
            end - overlap_tokens
        )

    return pieces


def split_by_sentences(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    pieces: List[str] = []

    current: List[str] = []

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_tokens = count_tokens(
            sentence
        )

        # ----------------------------------------------------
        # One sentence itself is too large.
        # ----------------------------------------------------

        if sentence_tokens > max_tokens:

            if current:

                pieces.append(
                    " ".join(current)
                )

                current = []

            pieces.extend(
                split_by_token_window(
                    sentence,
                    max_tokens,
                    overlap_tokens,
                )
            )

            continue

        candidate = " ".join(
            current + [sentence]
        )

        if (
            current
            and count_tokens(candidate)
            > max_tokens
        ):

            pieces.append(
                " ".join(current)
            )

            current = [
                sentence
            ]

        else:

            current.append(
                sentence
            )

    if current:

        pieces.append(
            " ".join(current)
        )

    return pieces


def split_oversized_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:

    if count_tokens(text) <= max_tokens:

        return [
            text.strip()
        ]

    pieces = split_by_sentences(
        text,
        max_tokens,
        overlap_tokens,
    )

    if (
        pieces
        and all(
            count_tokens(piece)
            <= max_tokens
            for piece in pieces
        )
    ):

        return pieces

    return split_by_token_window(
        text,
        max_tokens,
        overlap_tokens,
    )


# ============================================================
# SECTION HELPERS
# ============================================================

def node_units(
    node: SectionNode,
    include_children: bool = True,
) -> List[MarkdownUnit]:

    units: List[MarkdownUnit] = []

    if node.heading:

        units.append(
            node.heading
        )

    units.extend(
        node.content
    )

    if include_children:

        for child in node.children:

            units.extend(
                node_units(
                    child,
                    include_children=True,
                )
            )

    return units


def node_has_real_content(
    node: SectionNode,
) -> bool:

    return bool(
        node.content
        or node.children
    )


# ============================================================
# STRUCTURAL HEADING HANDLING
# ============================================================

def attach_structural_heading(
    heading: MarkdownUnit,
    chunks: List[Chunk],
    max_tokens: int,
    overlap_tokens: int,
) -> List[Chunk]:

    if not chunks:
        return []

    heading_text = heading.text.strip()

    if not heading_text:
        return chunks

    first = chunks[0]

    combined_text = (
        heading_text
        + "\n\n"
        + first.text.strip()
    )

    # --------------------------------------------------------
    # Best case: entire combined chunk fits.
    # --------------------------------------------------------

    if count_tokens(combined_text) <= max_tokens:

        first.text = combined_text

        first.token_count = (
            count_tokens(
                combined_text
            )
        )

        first.page_no, first.page_end = (
            combine_page_ranges(
                [
                    heading,
                    MarkdownUnit(
                        unit_type="paragraph",
                        text=first.text,
                        page_no=first.page_no,
                        page_end=first.page_end,
                    ),
                ]
            )
        )

        return chunks

    # --------------------------------------------------------
    # Reserve tokens for the structural heading.
    # --------------------------------------------------------

    heading_tokens = count_tokens(
        heading_text
    )

    if heading_tokens >= max_tokens:

        pieces = split_by_token_window(
            combined_text,
            max_tokens,
            overlap_tokens,
        )

        rebuilt: List[Chunk] = []

        for piece in pieces:

            rebuilt.append(
                Chunk(
                    text=piece,
                    page_no=first.page_no,
                    page_end=first.page_end,
                    section_path=(
                        first.section_path.copy()
                    ),
                    table_ref=(
                        first.table_ref.copy()
                    ),
                    token_count=count_tokens(
                        piece
                    ),
                )
            )

        return (
            rebuilt
            + chunks[1:]
        )

    available_tokens = (
        max_tokens
        - heading_tokens
    )

    content_pieces = split_oversized_text(
        first.text,
        available_tokens,
        overlap_tokens,
    )

    rebuilt: List[Chunk] = []

    for index, piece in enumerate(
        content_pieces
    ):

        if index == 0:

            text = (
                heading_text
                + "\n\n"
                + piece
            )

        else:

            text = piece

        rebuilt.append(
            Chunk(
                text=text,
                page_no=first.page_no,
                page_end=first.page_end,
                section_path=(
                    first.section_path.copy()
                ),
                table_ref=(
                    first.table_ref.copy()
                ),
                token_count=count_tokens(
                    text
                ),
            )
        )

    return (
        rebuilt
        + chunks[1:]
    )


# ============================================================
# STRUCTURE-AWARE CHUNKER
# ============================================================

class StructureAwareChunker:

    def __init__(
        self,
        max_tokens: int,
        overlap_tokens: int,
    ):

        if max_tokens <= 0:

            raise ValueError(
                "max_tokens must be positive."
            )

        if (
            overlap_tokens < 0
            or overlap_tokens >= max_tokens
        ):

            raise ValueError(
                "overlap_tokens must be >= 0 "
                "and smaller than max_tokens."
            )

        self.max_tokens = max_tokens
        self.overlap_tokens = (
            overlap_tokens
        )

    def create_chunks(
        self,
        root: SectionNode,
    ) -> List[Chunk]:

        chunks: List[Chunk] = []

        # ----------------------------------------------------
        # Content before first heading
        # ----------------------------------------------------

        if root.content:

            chunks.extend(
                self._chunk_content_group(
                    root.content,
                    None,
                )
            )

        # ----------------------------------------------------
        # Top-level sections
        # ----------------------------------------------------

        chunks.extend(
            self._chunk_section_children(
                root
            )
        )

        return [
            chunk
            for chunk in chunks
            if chunk.text.strip()
        ]

    def _chunk_section_children(
        self,
        parent: SectionNode,
    ) -> List[Chunk]:

        output: List[Chunk] = []

        pending_structural_headings: List[
            MarkdownUnit
        ] = []

        for child in parent.children:

            if (
                child.heading is not None
                and not child.content
                and not child.children
            ):

                pending_structural_headings.append(
                    child.heading
                )

                continue

            child_chunks = self._chunk_section(
                child
            )

            if (
                pending_structural_headings
                and child_chunks
            ):

                for heading in reversed(
                    pending_structural_headings
                ):

                    child_chunks = (
                        attach_structural_heading(
                            heading=heading,
                            chunks=child_chunks,
                            max_tokens=self.max_tokens,
                            overlap_tokens=self.overlap_tokens,
                        )
                    )

                pending_structural_headings = []

            output.extend(
                child_chunks
            )

        return output

    def _chunk_section(
        self,
        node: SectionNode,
    ) -> List[Chunk]:

        if (
            node.heading is not None
            and not node.content
            and not node.children
        ):

            return []

        whole_units = node_units(
            node,
            include_children=True,
        )

        whole = make_chunk_from_units(
            whole_units
        )

        if (
            whole is not None
            and whole.token_count
            <= self.max_tokens
        ):

            return [
                whole
            ]

        if not node.children:

            own_units: List[
                MarkdownUnit
            ] = []

            if node.heading:

                own_units.append(
                    node.heading
                )

            own_units.extend(
                node.content
            )

            return self._chunk_content_group(
                own_units,
                node.heading,
            )

        output: List[Chunk] = []

        if node.content:

            intro_units: List[
                MarkdownUnit
            ] = []

            if node.heading:

                intro_units.append(
                    node.heading
                )

            intro_units.extend(
                node.content
            )

            output.extend(
                self._chunk_content_group(
                    intro_units,
                    node.heading,
                )
            )

        child_chunks: List[
            Chunk
        ] = []

        pending_structural_headings: List[
            MarkdownUnit
        ] = []

        for child in node.children:

            if (
                child.heading is not None
                and not child.content
                and not child.children
            ):

                pending_structural_headings.append(
                    child.heading
                )

                continue

            current_child_chunks = (
                self._chunk_section(
                    child
                )
            )

            if (
                pending_structural_headings
                and current_child_chunks
            ):

                for heading in reversed(
                    pending_structural_headings
                ):

                    current_child_chunks = (
                        attach_structural_heading(
                            heading=heading,
                            chunks=current_child_chunks,
                            max_tokens=self.max_tokens,
                            overlap_tokens=self.overlap_tokens,
                        )
                    )

                pending_structural_headings = []

            child_chunks.extend(
                current_child_chunks
            )

        if (
            node.heading
            and child_chunks
            and not node.content
        ):

            child_chunks = (
                self._attach_parent_heading(
                    node.heading,
                    child_chunks,
                )
            )

        output.extend(
            child_chunks
        )

        return [
            chunk
            for chunk in output
            if chunk.text.strip()
        ]

    def _attach_parent_heading(
        self,
        heading: MarkdownUnit,
        chunks: List[Chunk],
    ) -> List[Chunk]:

        return attach_structural_heading(
            heading=heading,
            chunks=chunks,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )

    def _chunk_content_group(
        self,
        units: List[MarkdownUnit],
        heading: Optional[MarkdownUnit],
    ) -> List[Chunk]:

        if not units:
            return []

        whole = make_chunk_from_units(
            units,
            section_path_override=(
                heading.section_path.copy()
                if heading is not None
                else None
            ),
        )

        if (
            whole is not None
            and whole.token_count
            <= self.max_tokens
        ):

            return [
                whole
            ]

        output: List[Chunk] = []

        current: List[
            MarkdownUnit
        ] = []

        for unit in units:

            proposed = (
                current
                + [unit]
            )

            candidate = (
                make_chunk_from_units(
                    proposed,
                    section_path_override=(
                        heading.section_path.copy()
                        if heading is not None
                        else None
                    ),
                )
            )

            if (
                candidate is not None
                and candidate.token_count
                <= self.max_tokens
            ):

                current = proposed

                continue

            if (
                len(current) == 1
                and current[0].unit_type
                == "heading"
                and unit.unit_type
                != "heading"
            ):

                output.extend(
                    self._split_unit_with_heading(
                        current[0],
                        unit,
                    )
                )

                current = []

                continue

            if current:

                current_chunk = (
                    make_chunk_from_units(
                        current,
                        section_path_override=(
                            heading.section_path.copy()
                            if heading is not None
                            else None
                        ),
                    )
                )

                if current_chunk is not None:

                    output.append(
                        current_chunk
                    )

                current = []

            if unit.unit_type == "heading":

                current = [
                    unit
                ]

                continue

            output.extend(
                self._split_single_unit(
                    unit
                )
            )

        if current:

            candidate = (
                make_chunk_from_units(
                    current,
                    section_path_override=(
                        heading.section_path.copy()
                        if heading is not None
                        else None
                    ),
                )
            )

            if (
                candidate is not None
                and candidate.token_count
                <= self.max_tokens
            ):

                output.append(
                    candidate
                )

            else:

                output.extend(
                    self._split_units_tokenwise(
                        current
                    )
                )

        return [
            chunk
            for chunk in output
            if chunk.text.strip()
        ]

    def _split_unit_with_heading(
        self,
        heading: MarkdownUnit,
        unit: MarkdownUnit,
    ) -> List[Chunk]:

        heading_text = (
            heading.text.strip()
        )

        heading_tokens = count_tokens(
            heading_text
        )

        if heading_tokens >= self.max_tokens:

            return self._split_units_tokenwise(
                [
                    heading,
                    unit,
                ]
            )

        available_tokens = (
            self.max_tokens
            - heading_tokens
        )

        content_pieces = (
            split_oversized_text(
                unit.text,
                available_tokens,
                self.overlap_tokens,
            )
        )

        output: List[Chunk] = []

        page_no, page_end = (
            combine_page_ranges(
                [
                    heading,
                    unit,
                ]
            )
        )

        for index, piece in enumerate(
            content_pieces
        ):

            if index == 0:

                text = (
                    heading_text
                    + "\n\n"
                    + piece
                )

            else:

                text = piece

            output.append(
                Chunk(
                    text=text,
                    page_no=page_no,
                    page_end=page_end,
                    section_path=(
                        unit.section_path.copy()
                    ),
                    table_ref=(
                        [unit.table_ref]
                        if unit.table_ref
                        else []
                    ),
                    token_count=count_tokens(
                        text
                    ),
                )
            )

        return output

    def _split_single_unit(
        self,
        unit: MarkdownUnit,
    ) -> List[Chunk]:

        pieces = split_oversized_text(
            unit.text,
            self.max_tokens,
            self.overlap_tokens,
        )

        return [
            Chunk(
                text=piece,
                page_no=unit.page_no,
                page_end=unit.page_end,
                section_path=(
                    unit.section_path.copy()
                ),
                table_ref=(
                    [unit.table_ref]
                    if unit.table_ref
                    else []
                ),
                token_count=count_tokens(
                    piece
                ),
            )
            for piece in pieces
            if piece.strip()
        ]

    def _split_units_tokenwise(
        self,
        units: List[MarkdownUnit],
    ) -> List[Chunk]:

        text = "\n\n".join(
            unit.text
            for unit in units
        )

        pieces = split_by_token_window(
            text,
            self.max_tokens,
            self.overlap_tokens,
        )

        page_no, page_end = (
            combine_page_ranges(
                units
            )
        )

        heading_units = [
            unit
            for unit in units
            if unit.unit_type == "heading"
        ]

        if heading_units:

            section_path = (
                heading_units[0]
                .section_path.copy()
            )

        else:

            section_path = (
                units[0].section_path.copy()
                if units
                else []
            )

        table_refs: List[str] = []

        for unit in units:

            if (
                unit.table_ref
                and unit.table_ref
                not in table_refs
            ):

                table_refs.append(
                    unit.table_ref
                )

        return [
            Chunk(
                text=piece,
                page_no=page_no,
                page_end=page_end,
                section_path=(
                    section_path.copy()
                ),
                table_ref=table_refs,
                token_count=count_tokens(
                    piece
                ),
            )
            for piece in pieces
            if piece.strip()
        ]


# ============================================================
# VALIDATION
# ============================================================

def is_heading_only_chunk(
    chunk: Chunk,
) -> bool:

    text = chunk.text.strip()

    if not text:
        return True

    blocks = split_markdown_into_blocks(
        text
    )

    if not blocks:
        return True

    return all(
        is_heading(block)
        for block in blocks
    )


def validate_chunks(
    chunks: List[Chunk],
    max_tokens: int,
) -> None:

    errors: List[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        actual_token_count = (
            count_tokens(
                chunk.text
            )
        )

        if actual_token_count > max_tokens:

            errors.append(
                f"Chunk {index} exceeds "
                f"MAX_CHUNK_TOKENS: "
                f"{actual_token_count} > "
                f"{max_tokens}"
            )

        if (
            chunk.token_count
            != actual_token_count
        ):

            errors.append(
                f"Chunk {index} has incorrect "
                f"token_count metadata."
            )

        if not chunk.text.strip():

            errors.append(
                f"Chunk {index} is empty."
            )

        if is_heading_only_chunk(
            chunk
        ):

            errors.append(
                f"Chunk {index} is a "
                f"heading-only/orphan chunk: "
                f"{chunk.text[:150]!r}"
            )

        if not isinstance(
            chunk.section_path,
            list,
        ):

            errors.append(
                f"Chunk {index} has invalid "
                f"section_path."
            )

        if not isinstance(
            chunk.table_ref,
            list,
        ):

            errors.append(
                f"Chunk {index} has invalid "
                f"table_ref."
            )

    if errors:

        message = (
            "\n"
            + "=" * 80
            + "\n"
            + "STAGE 2 VALIDATION FAILED"
            + "\n"
            + "=" * 80
            + "\n"
        )

        message += "\n".join(
            f"  - {error}"
            for error in errors
        )

        message += (
            "\n"
            + "=" * 80
        )

        raise RuntimeError(
            message
        )


# ============================================================
# SERIALIZATION
# ============================================================

def chunk_to_record(
    chunk: Chunk,
    document_name: str,
    document_id: str,
    chunk_id: str,
) -> dict:

    return {
        "document_name":
            document_name,

        "document_id":
            document_id,

        "chunk_id":
            chunk_id,

        "page_no":
            chunk.page_no,

        "page_end":
            chunk.page_end,

        "section_path":
            chunk.section_path,

        "table_ref":
            chunk.table_ref,

        "chunk_text":
            chunk.text,

        "token_count":
            chunk.token_count,
    }


# ============================================================
# PROCESS ONE DOCUMENT
# ============================================================

def process_document(
    markdown_file: Path,
    manifest_file: Path,
    output_handle,
    chunker: StructureAwareChunker,
) -> dict:

    print("\n" + "=" * 80)

    print(
        f"Processing document: "
        f"{markdown_file.name}"
    )

    # --------------------------------------------------------
    # Load Markdown
    # --------------------------------------------------------

    markdown = markdown_file.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    with manifest_file.open(
        "r",
        encoding="utf-8",
    ) as f:

        manifest = json.load(f)

    document_id = (
        markdown_file.stem
    )

    document_name = (
        manifest.get("document_name")
        or f"{document_id}.pdf"
    )

    # --------------------------------------------------------
    # Build page mapper
    # --------------------------------------------------------

    page_mapper = ManifestPageMapper(
        manifest
    )

    # --------------------------------------------------------
    # Parse Markdown
    # --------------------------------------------------------

    units = build_markdown_units(
        markdown,
        page_mapper,
    )

    # --------------------------------------------------------
    # Build section tree
    # --------------------------------------------------------

    root = build_section_tree(
        units
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = chunker.create_chunks(
        root
    )

    # --------------------------------------------------------
    # HARD VALIDATION BEFORE WRITING
    # --------------------------------------------------------

    validate_chunks(
        chunks,
        chunker.max_tokens,
    )

    print(
        f"  Markdown units : "
        f"{len(units)}"
    )

    print(
        f"  Chunks         : "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Metadata checks
    # --------------------------------------------------------

    missing_pages = 0
    missing_table_pages = 0

    for chunk in chunks:

        if (
            chunk.page_no is None
            or chunk.page_end is None
        ):

            missing_pages += 1

        if (
            chunk.table_ref
            and (
                chunk.page_no is None
                or chunk.page_end is None
            )
        ):

            missing_table_pages += 1

    if missing_pages:

        print(
            f"  WARNING: "
            f"{missing_pages} chunk(s) "
            f"have missing page metadata"
        )

    else:

        print(
            "  Page-metadata check: PASS"
        )

    if missing_table_pages:

        print(
            f"  WARNING: "
            f"{missing_table_pages} table "
            f"chunk(s) have missing page "
            f"metadata"
        )

    # --------------------------------------------------------
    # Write chunks
    # --------------------------------------------------------

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        chunk_id = (
            f"{document_id}_chunk_"
            f"{index:04d}"
        )

        record = chunk_to_record(
            chunk=chunk,
            document_name=document_name,
            document_id=document_id,
            chunk_id=chunk_id,
        )

        output_handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    token_counts = [
        chunk.token_count
        for chunk in chunks
    ]

    return {
        "document_id":
            document_id,

        "document_name":
            document_name,

        "source_markdown":
            str(markdown_file),

        "source_manifest":
            str(manifest_file),

        "unit_count":
            len(units),

        "chunk_count":
            len(chunks),

        "min_chunk_tokens":
            (
                min(token_counts)
                if token_counts
                else 0
            ),

        "max_chunk_tokens":
            (
                max(token_counts)
                if token_counts
                else 0
            ),

        "avg_chunk_tokens":
            (
                round(
                    sum(token_counts)
                    / len(token_counts),
                    2,
                )
                if token_counts
                else 0
            ),

        "oversized_chunk_count":
            0,

        "missing_page_metadata_chunks":
            missing_pages,

        "missing_table_page_metadata_chunks":
            missing_table_pages,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)

    print(
        "STAGE 2 - "
        "STRUCTURE-AWARE HIERARCHICAL "
        "CHUNKING"
    )

    print("=" * 80)

    print(
        f"Token-count model : "
        f"{TOKEN_COUNT_MODEL}"
    )

    print(
        f"tiktoken encoding : "
        f"{ENCODING.name}"
    )

    print(
        f"Max chunk tokens  : "
        f"{MAX_CHUNK_TOKENS}"
    )

    print(
        f"Overlap tokens    : "
        f"{OVERLAP_TOKENS}"
    )

    print(
        f"Documents dir     : "
        f"{DOCUMENTS_DIR}"
    )

    print(
        f"Manifests dir     : "
        f"{MANIFESTS_DIR}"
    )

    print(
        f"Output file       : "
        f"{CHUNKS_FILE}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not DOCUMENTS_DIR.exists():

        raise FileNotFoundError(
            f"Documents directory does not exist:\n"
            f"{DOCUMENTS_DIR}"
        )

    if not MANIFESTS_DIR.exists():

        raise FileNotFoundError(
            f"Manifests directory does not exist:\n"
            f"{MANIFESTS_DIR}"
        )

    if (
        OVERLAP_TOKENS
        >= MAX_CHUNK_TOKENS
    ):

        raise ValueError(
            "OVERLAP_TOKENS must be smaller "
            "than MAX_CHUNK_TOKENS."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # DISCOVER MARKDOWN FILES
    # ========================================================

    markdown_files = sorted(
        DOCUMENTS_DIR.rglob(
            "*.md"
        )
    )

    if not markdown_files:

        raise FileNotFoundError(
            f"No Markdown files found under:\n"
            f"{DOCUMENTS_DIR}"
        )

    print(
        f"\nFound "
        f"{len(markdown_files)} "
        f"Markdown file(s)."
    )

    # ========================================================
    # CHUNKER
    # ========================================================

    chunker = StructureAwareChunker(
        max_tokens=MAX_CHUNK_TOKENS,
        overlap_tokens=OVERLAP_TOKENS,
    )

    all_statistics: List[dict] = []

    # ========================================================
    # WRITE JSONL
    # ========================================================

    with CHUNKS_FILE.open(
        "w",
        encoding="utf-8",
    ) as output_handle:

        for markdown_file in markdown_files:

            manifest_file = (
                discover_manifest(
                    markdown_file,
                    MANIFESTS_DIR,
                )
            )

            statistics = (
                process_document(
                    markdown_file=markdown_file,
                    manifest_file=manifest_file,
                    output_handle=output_handle,
                    chunker=chunker,
                )
            )

            all_statistics.append(
                statistics
            )

    # ========================================================
    # STAGE 2 STATISTICS
    # ========================================================

    total_chunks = sum(
        item["chunk_count"]
        for item in all_statistics
    )

    total_units = sum(
        item["unit_count"]
        for item in all_statistics
    )

    oversized_chunks = sum(
        item["oversized_chunk_count"]
        for item in all_statistics
    )

    missing_page_chunks = sum(
        item[
            "missing_page_metadata_chunks"
        ]
        for item in all_statistics
    )

    missing_table_page_chunks = sum(
        item[
            "missing_table_page_metadata_chunks"
        ]
        for item in all_statistics
    )

    stage2_manifest = {

        "schema_version":
            "1.5",

        "stage":
            "stage_2_structure_aware_chunking",

        "chunking_strategy":
            "structure_aware_hierarchical",

        "token_count_model":
            TOKEN_COUNT_MODEL,

        "tokenizer_encoding":
            ENCODING.name,

        "max_chunk_tokens":
            MAX_CHUNK_TOKENS,

        "overlap_tokens":
            OVERLAP_TOKENS,

        "document_count":
            len(markdown_files),

        "total_markdown_units":
            total_units,

        "total_chunks":
            total_chunks,

        "oversized_chunks":
            oversized_chunks,

        "chunks_with_missing_page_metadata":
            missing_page_chunks,

        "table_chunks_with_missing_page_metadata":
            missing_table_page_chunks,

        "heading_only_chunks_removed":
            True,

        "heading_only_headings_preserved_as_context":
            True,

        "documents":
            all_statistics,
    }

    with STAGE2_MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            stage2_manifest,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "STAGE 2 COMPLETE"
    )

    print("=" * 80)

    print(
        f"Documents processed : "
        f"{len(markdown_files)}"
    )

    print(
        f"Total chunks        : "
        f"{total_chunks}"
    )

    print(
        f"Oversized chunks    : "
        f"{oversized_chunks}"
    )

    print(
        f"Missing page data   : "
        f"{missing_page_chunks}"
    )

    print(
        f"Missing table pages : "
        f"{missing_table_page_chunks}"
    )

    print(
        "\nStructural validation: PASS"
    )

    print(
        "\nChunks written to:\n"
        f"{CHUNKS_FILE}"
    )

    print(
        "\nStage 2 manifest:\n"
        f"{STAGE2_MANIFEST_FILE}"
    )


if __name__ == "__main__":
    main()