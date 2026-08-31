"""
STAGE 1 - DOCLING DOCUMENT INGESTION
====================================

Responsibilities
----------------
1. Read PDF documents from INPUT_DIR.
2. Convert PDFs with Docling.
3. Apply docling-hierarchical-pdf to infer/correct heading hierarchy.
4. Preserve Docling structural elements and original PDF provenance.
5. Export the complete document to Markdown.
6. Extract tables directly from Docling TableItem objects.
7. Save every original table as a separate Markdown file.
8. Generate an AI retrieval-oriented summary for every table.
9. Replace the original table in the main Markdown with:
       - table summary
       - original PDF page information
       - reference to the separately stored original table
10. Save a provenance manifest containing:
       - document metadata
       - element order
       - element type
       - Docling self_ref
       - semantic label
       - hierarchy level
       - text
       - original PDF page_start/page_end
       - detailed provenance (bbox/charspan)
       - table references
11. Save processing statistics.

IMPORTANT
---------
The provenance manifest is the authoritative source for original
PDF page numbers.

The Markdown document must NOT be used later to infer PDF pages.

Stage 2 will consume:
    - manifest
    - main Markdown
    - original table Markdown files

The original table is NEVER lost. The main Markdown contains a
retrieval-friendly summary while the complete table is stored separately.

HEADING HIERARCHY
-----------------
Native Docling HeadingHierarchyOptions are intentionally NOT used.

We previously tested Docling's native heading hierarchy inference
and observed incorrect/inconsistent heading levels.

Therefore Stage 1 now uses:

    Docling conversion
        ->
    docling-hierarchical-pdf ResultPostprocessor
        ->
    corrected DoclingDocument
        ->
    Markdown export

The postprocessor is applied BEFORE:
    - element extraction
    - Markdown export
    - table processing

This ensures that both the exported Markdown and the provenance
manifest see the corrected Docling hierarchy.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool  # CRITICAL FOR SYSTEM-LEVEL MEMORY RECOVERY
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
# ============================================================
# IMPORTANT:
# These environment settings MUST happen before importing
# torch / Docling.
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"        # Restrict CPU thread splitting allocation
os.environ["MKL_NUM_THREADS"] = "1"


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv

load_dotenv()

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import TextItem, TableItem

# ------------------------------------------------------------
# DOCLING-HIERARCHICAL-PDF
# ------------------------------------------------------------
# IMPORTANT:
#
# The PyPI distribution is:
#
#     docling-hierarchical-pdf
#
# But the Python import is:
#
#     hierarchical.postprocessor
#
# ResultPostprocessor operates on the Docling ConversionResult
# and modifies result.document in place.
# ------------------------------------------------------------

from hierarchical.postprocessor import ResultPostprocessor


# ------------------------------------------------------------
# OPENAI IMPORT
# ------------------------------------------------------------
# We are NOT using the OpenAI API.
#
# The OpenAI Python client is intentionally retained because
# Groq provides an OpenAI-compatible API endpoint.
#
# The client connects to:
# https://api.groq.com/openai/v1
# ------------------------------------------------------------

from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

# INPUT_DIR = Path(
#     r"D:/VECTORDBCREATION/document_processing_db_creation/Files"
# )

# OUTPUT_DIR = Path(
#     r"D:/VECTORDBCREATION/document_processing_db_creation/"
#     r"output_stage_1_doc_ingestion"
# )

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = PROJECT_ROOT / "data" / "input"

OUTPUT_DIR = PROJECT_ROOT / "data" / "stage_1"


# ------------------------------------------------------------
# OPENAI CONFIGURATION - DISABLED FOR NOW
# ------------------------------------------------------------

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_MODEL = "gpt-4o-mini"


# ------------------------------------------------------------
# GROQ CONFIGURATION
# ------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Fast production model currently available on Groq.
GROQ_MODEL = "openai/gpt-oss-20b"


SUPPORTED_EXTENSIONS = {
    ".pdf"
}


# ============================================================
# HELPERS
# ============================================================

def get_document_page_count(
    document: Any,
) -> Optional[int]:
    """
    Safely obtain the total number of pages from the Docling document.

    Docling versions may expose num_pages as either a property or
    a callable method. Never allow the callable itself to enter
    the JSON manifest.
    """

    value = getattr(
        document,
        "num_pages",
        None,
    )

    if value is None:
        return None

    if callable(value):
        value = value()

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sanitize_filename(
    name: str,
) -> str:
    """
    Convert a filename into a filesystem-safe identifier.
    """

    name = Path(name).stem

    name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        name,
    )

    name = re.sub(
        r"_+",
        "_",
        name,
    )

    name = name.strip("_")

    return name[:150]


def get_page_range(
    item: Any,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract original PDF page range directly from Docling provenance.

    This is the ONLY source used for original PDF page information.

    Returns
    -------
    (page_start, page_end)
    """

    provenance = getattr(
        item,
        "prov",
        None,
    )

    if not provenance:
        return None, None

    pages: List[int] = []

    for prov in provenance:

        page_no = getattr(
            prov,
            "page_no",
            None,
        )

        if page_no is not None:
            pages.append(int(page_no))

    if not pages:
        return None, None

    return min(pages), max(pages)


def get_item_label(
    item: Any,
) -> Optional[str]:
    """
    Safely obtain Docling's semantic label.
    """

    label = getattr(
        item,
        "label",
        None,
    )

    if label is None:
        return None

    value = getattr(
        label,
        "value",
        None,
    )

    if value is not None:
        return str(value)

    return str(label)


def serialize_bbox(
    bbox: Any,
) -> Optional[Dict[str, Any]]:
    """
    Convert Docling BoundingBox into JSON-safe data.
    """

    if bbox is None:
        return None

    result: Dict[str, Any] = {}

    for field in (
        "l",
        "t",
        "r",
        "b",
        "coord_origin",
    ):

        value = getattr(
            bbox,
            field,
            None,
        )

        if value is not None:

            if field == "coord_origin":
                result[field] = str(value)

            else:
                result[field] = float(value)

    return result


def extract_provenance(
    item: Any,
) -> List[Dict[str, Any]]:
    """
    Preserve detailed Docling provenance.

    Page numbers remain the primary provenance information.
    Bounding boxes and charspans are retained for future use.
    """

    provenance = getattr(
        item,
        "prov",
        None,
    )

    if not provenance:
        return []

    result: List[Dict[str, Any]] = []

    for prov in provenance:

        result.append(
            {
                "page_no": getattr(
                    prov,
                    "page_no",
                    None,
                ),
                "bbox": serialize_bbox(
                    getattr(
                        prov,
                        "bbox",
                        None,
                    )
                ),
                "charspan": list(
                    getattr(
                        prov,
                        "charspan",
                        (),
                    )
                ),
            }
        )

    return result


def get_text(
    item: Any,
) -> Optional[str]:
    """
    Safely obtain textual content from a Docling item.
    """

    text = getattr(
        item,
        "text",
        None,
    )

    if text is None:
        return None

    return str(text)


# ============================================================
# MAIN INGESTION CLASS
# ============================================================

class DoclingIngestion:

    def __init__(
        self,
        api_key: str,
        model: str = GROQ_MODEL,
    ):

        if not api_key:

            raise RuntimeError(
                "GROQ_API_KEY is not set.\n\n"
                "Windows PowerShell:\n"
                "$env:GROQ_API_KEY='your-key'\n\n"
                "Windows CMD:\n"
                "set GROQ_API_KEY=your-key"
            )

        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------
        # Groq provides an OpenAI-compatible API.
        #
        # Therefore we can continue using the existing
        # OpenAI Python client without rewriting the rest
        # of the application.
        # ----------------------------------------------------

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

        self.model = model

        # ----------------------------------------------------
        # DOCLING CONFIGURATION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # HeadingHierarchyOptions is intentionally NOT configured.
        #
        # Heading hierarchy correction is now handled by:
        #
        #     docling-hierarchical-pdf
        #
        # after Docling conversion.
        # ----------------------------------------------------

        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                num_threads=1,
                device="cpu",
            ),
            do_ocr=False,
            do_table_structure=True,

            # Required for style-based hierarchy inference
            # used by the hierarchical postprocessor.
            generate_parsed_pages=True,
        )

        # self.converter = DocumentConverter(
        #     format_options={
        #         "pdf": PdfFormatOption(
        #             pipeline_options=pipeline_options
        #         )
        #     }
        # )

        self.converter = DocumentConverter(
                    format_options={
                         InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend
                )
                    }
        )

        self.stats = {
            "files_found": 0,
            "files_processed": 0,
            "files_failed": 0,
            "tables_found": 0,
            "tables_processed": 0,

            # ------------------------------------------------
            # Renamed from openai_calls because the provider
            # is now Groq.
            # ------------------------------------------------

            "llm_calls": 0,
            "total_tokens": 0,

            "errors": [],
        }

    # ========================================================
    # FILE DISCOVERY
    # ========================================================

    def find_input_files(
        self,
        input_dir: Path,
    ) -> List[Path]:

        if not input_dir.exists():

            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{input_dir}"
            )

        files = [
            p
            for p in input_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]

        return sorted(files)

    # ========================================================
    # DOCUMENT ID
    # ========================================================

    def create_document_id(
        self,
        file_path: Path,
    ) -> str:

        return sanitize_filename(
            file_path.name
        )

    # ========================================================
    # APPLY HEADING HIERARCHY
    # ========================================================

    def apply_heading_hierarchy(
        self,
        result: Any,
        source_path: Path,
    ) -> None:
        """
        Apply docling-hierarchical-pdf to the Docling
        ConversionResult.

        IMPORTANT:
        This is performed BEFORE:
            - element extraction
            - Markdown export
            - table processing

        ResultPostprocessor modifies result.document in place.

        The PDF source path is explicitly supplied so that the
        postprocessor can access PDF metadata/bookmarks when
        required.
        """

        print(
            "     Applying "
            "docling-hierarchical-pdf..."
        )

        postprocessor = ResultPostprocessor(
            result,
            source=str(source_path),
        )

        postprocessor.process()

        print(
            "     ✓ Heading hierarchy "
            "postprocessing complete."
        )

    # ========================================================
    # EXTRACT STRUCTURED ELEMENTS
    # ========================================================

    def extract_elements(
        self,
        document: Any,
    ) -> List[Dict[str, Any]]:
        """
        Extract Docling elements in document reading order.

        These are SOURCE ELEMENTS, not final RAG chunks.

        Every element retains its original PDF provenance.

        Tables are represented explicitly and linked through
        their Docling self_ref.

        IMPORTANT:
        This method is called AFTER the hierarchy postprocessor,
        so hierarchy_level reflects the corrected Docling
        document structure.
        """

        elements: List[Dict[str, Any]] = []

        for index, result in enumerate(
            document.iterate_items()
        ):

            # Docling returns:
            #
            #     (item, hierarchy_level)

            item, level = result

            page_start, page_end = (
                get_page_range(item)
            )

            item_type = type(item).__name__

            label = get_item_label(
                item
            )

            self_ref = getattr(
                item,
                "self_ref",
                None,
            )

            record: Dict[str, Any] = {

                "element_index": index,

                "element_type": item_type,

                "label": label,

                "self_ref": self_ref,

                "hierarchy_level": level,

                "page_start": page_start,

                "page_end": page_end,

                "provenance": extract_provenance(
                    item
                ),
            }

            # ------------------------------------------------
            # TEXT ELEMENT
            # ------------------------------------------------

            if isinstance(
                item,
                TextItem,
            ):

                record["text"] = (
                    get_text(item)
                    or ""
                )

            # ------------------------------------------------
            # TABLE ELEMENT
            # ------------------------------------------------

            elif isinstance(
                item,
                TableItem,
            ):

                record["text"] = ""

                record["table"] = True

                record["table_ref"] = self_ref

            # ------------------------------------------------
            # OTHER ELEMENT
            # ------------------------------------------------

            else:

                text = get_text(item)

                if text:
                    record["text"] = text

            elements.append(
                record
            )

        return elements

    # ========================================================
    # TABLE EXTRACTION
    # ========================================================

    def extract_table_records(
        self,
        document: Any,
        elements: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract tables directly from Docling's document model.

        We DO NOT parse Markdown tables.

        This guarantees that:
            TableItem
            -> TableData
            -> provenance
            -> original table

        remain associated.
        """

        tables = list(
            document.tables
        )

        self.stats["tables_found"] += len(
            tables
        )

        # ----------------------------------------------------
        # Map Docling self_ref -> element index.
        # ----------------------------------------------------

        element_index_by_ref = {
            element.get("self_ref"): element.get(
                "element_index"
            )
            for element in elements
            if element.get("self_ref")
        }

        records: List[Dict[str, Any]] = []

        for table_number, table in enumerate(
            tables,
            start=1,
        ):

            page_start, page_end = (
                get_page_range(table)
            )

            table_ref = getattr(
                table,
                "self_ref",
                None,
            )

            table_markdown = (
                table.export_to_markdown(
                    doc=document
                )
            )

            element_index = (
                element_index_by_ref.get(
                    table_ref
                )
            )

            records.append(
                {
                    "table_number": table_number,

                    "element_index": element_index,

                    "page_start": page_start,

                    "page_end": page_end,

                    "docling_ref": table_ref,

                    "markdown": table_markdown,

                    "provenance": extract_provenance(
                        table
                    ),
                }
            )

        return records

    # ========================================================
    # TABLE SUMMARY
    # ========================================================

    def generate_table_summary(
        self,
        table_markdown: str,
        table_number: int,
        page_start: Optional[int],
        page_end: Optional[int],
        context: str,
    ) -> str:
        """
        Generate a retrieval-oriented summary.

        The summary is NOT a replacement for the original table.

        The original table is always stored separately.
        """

        if page_start is not None:

            if (
                page_end is not None
                and page_end != page_start
            ):

                page_text = (
                    f"Original PDF pages: "
                    f"{page_start}-{page_end}"
                )

            else:

                page_text = (
                    f"Original PDF page: "
                    f"{page_start}"
                )

        else:

            page_text = (
                "Original PDF page: unavailable"
            )

        sysprompt = """
You are a professional data analyst who creates concise and systematic summaries of tabular data.

STRICT GUIDELINES:

1. If the table includes years as a column or row,
   summarize data for ALL years in their natural order.
   DO NOT skip any year.

2. Summaries must be comprehensive and concise.

3. Cover every important piece of information in the table.
   Do not skip important columns, rows, categories,
   financial metrics, or data points.

4. Report data in the order it appears in the table.

5. Preserve exact numbers, percentages, dates, years,
   entity names and financial terminology.

6. Do not invent missing values.

7. Do not infer relationships that are not visible in the table.

8. The output will be used for retrieval in a RAG system,
   so preserve terminology that users may search for.
"""

        prompt = f"""
You are analyzing a table from a business document and generating
a clear, comprehensive summary.

Table number: {table_number}

SURROUNDING DOCUMENT CONTEXT:
{context}

TABLE:
{table_markdown}

Create an accurate, information-rich summary that includes:

**Purpose**: What this table shows (1 sentence)

**Key Information**:
- Important data points or values
- Key categories or classifications
- Notable figures or percentages

**Notable Insights**:
- Patterns, trends, or relationships in the data
- Any significant observations or conclusions

Use bullet points where appropriate for clarity.

The summary must identify:

1. What the table represents.
2. The entities/categories represented.
3. Important metrics and measures.
4. Years/date ranges if present.
5. Important numerical values, percentages, totals,
   ranges or comparisons.
6. Important relationships between rows and columns.
7. Trends or notable observations that are explicitly
   supported by the table.

Rules:

- Do not invent information.
- Do not make unsupported conclusions.
- Preserve important numbers.
- Preserve important years.
- Preserve important entity/category names.
- Do not omit important financial metrics.
- Do not claim relationships that are not visible in the table.
- Keep the summary concise but sufficiently detailed.
"""

        # ----------------------------------------------------
        # GROQ / OpenAI-compatible API
        # ----------------------------------------------------

        response = (
            self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": sysprompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.1,
                max_completion_tokens=2000,
            )
        )

        self.stats["llm_calls"] += 1

        if response.usage:

            self.stats["total_tokens"] += (
                response.usage.total_tokens
            )

        content = (
            response.choices[0]
            .message
            .content
        )

        # content = "summary placeholder"

        if not content:

            raise RuntimeError(
                f"Groq returned an empty summary "
                f"for table {table_number}."
            )

        return content.strip()

    # ========================================================
    # TABLE CONTEXT
    # ========================================================

    def get_table_context(
        self,
        original_markdown: str,
        table_markdown: str,
        max_chars: int = 300,
    ) -> str:
        """
        Obtain surrounding context from the ORIGINAL Markdown.

        IMPORTANT:
        This method is used only for table-summary generation.

        It is NOT used for provenance.
        """

        if not table_markdown:
            return ""

        position = original_markdown.find(
            table_markdown
        )

        if position == -1:
            return ""

        start = max(
            0,
            position - max_chars,
        )

        end = min(
            len(original_markdown),
            position
            + len(table_markdown)
            + max_chars,
        )

        context = original_markdown[
            start:end
        ]

        context = context.replace(
            table_markdown,
            "",
            1,
        )

        return context.strip()

    # ========================================================
    # SAVE ORIGINAL TABLE
    # ========================================================

    def save_table(
        self,
        output_dir: Path,
        document_id: str,
        document_name: str,
        table_number: int,
        table_markdown: str,
        page_start: Optional[int],
        page_end: Optional[int],
        docling_ref: Optional[str],
    ) -> str:

        tables_dir = (
            output_dir / "tables"
        )

        tables_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        filename = (
            f"{document_id}"
            f"_table_{table_number:03d}.md"
        )

        table_path = (
            tables_dir / filename
        )

        if page_start is not None:

            if (
                page_end is not None
                and page_end != page_start
            ):

                page_text = (
                    f"**Source PDF pages:** "
                    f"{page_start}-{page_end}\n\n"
                )

            else:

                page_text = (
                    f"**Source PDF page:** "
                    f"{page_start}\n\n"
                )

        else:

            page_text = (
                "**Source PDF page:** unavailable\n\n"
            )

        ref_text = ""

        if docling_ref:
            ref_text = (
                f"**Docling reference:** "
                f"`{docling_ref}`\n\n"
            )

        content = (
            f"# Table {table_number}\n\n"
            f"**Source document:** "
            f"{document_name}\n\n"
            f"{page_text}"
            f"{ref_text}"
            f"## Original Table\n\n"
            f"{table_markdown}\n"
        )

        table_path.write_text(
            content,
            encoding="utf-8",
        )

        return (
            f"tables/{filename}"
        )

    # ========================================================
    # BUILD TABLE REPLACEMENT
    # ========================================================

    def build_table_replacement(
        self,
        table_number: int,
        summary: str,
        table_path: str,
        page_start: Optional[int],
        page_end: Optional[int],
    ) -> str:

        if page_start is not None:

            if (
                page_end is not None
                and page_end != page_start
            ):

                page_text = (
                    f"**Source PDF pages:** "
                    f"{page_start}-{page_end}\n\n"
                )

            else:

                page_text = (
                    f"**Source PDF page:** "
                    f"{page_start}\n\n"
                )

        else:

            page_text = (
                "**Source PDF page:** unavailable\n\n"
            )

        return (
            f"$$ Table {table_number} — Summary\n\n"
            f"{page_text}"
            f"{summary}\n\n"
            f"**Original table:** "
            f"`{table_path}`\n\n"
        )

    # ========================================================
    # REPLACE TABLES IN MARKDOWN
    # ========================================================

    def replace_tables(
        self,
        original_markdown: str,
        table_records: List[Dict[str, Any]],
        document_id: str,
        document_name: str,
        output_dir: Path,
    ) -> Tuple[
        str,
        List[Dict[str, Any]]
    ]:
        """
        Replace original Docling-generated table Markdown
        with summary blocks.

        Tables are processed in Docling document order.

        Provenance ALWAYS comes from TableItem.prov.

        Markdown is used ONLY to locate the rendered table
        for replacement.
        """

        markdown = original_markdown

        replacement_records: List[
            Dict[str, Any]
        ] = []

        for table in table_records:

            table_number = table[
                "table_number"
            ]

            table_markdown = table[
                "markdown"
            ]

            page_start = table[
                "page_start"
            ]

            page_end = table[
                "page_end"
            ]

            docling_ref = table.get(
                "docling_ref"
            )

            element_index = table.get(
                "element_index"
            )

            if not table_markdown.strip():

                raise RuntimeError(
                    f"Table {table_number} "
                    f"({docling_ref}) produced "
                    f"empty Markdown."
                )

            # ------------------------------------------------
            # Context MUST come from the untouched document.
            # ------------------------------------------------

            context = (
                self.get_table_context(
                    original_markdown,
                    table_markdown,
                )
            )

            print(
                f"   Table {table_number}: "
                f"generating summary with Groq..."
            )

            summary = (
                self.generate_table_summary(
                    table_markdown=table_markdown,
                    table_number=table_number,
                    page_start=page_start,
                    page_end=page_end,
                    context=context,
                )
            )

            # ------------------------------------------------
            # Save complete original table.
            # ------------------------------------------------

            table_path = self.save_table(
                output_dir=output_dir,
                document_id=document_id,
                document_name=document_name,
                table_number=table_number,
                table_markdown=table_markdown,
                page_start=page_start,
                page_end=page_end,
                docling_ref=docling_ref,
            )

            # ------------------------------------------------
            # Build summary replacement.
            # ------------------------------------------------

            replacement = (
                self.build_table_replacement(
                    table_number=table_number,
                    summary=summary,
                    table_path=table_path,
                    page_start=page_start,
                    page_end=page_end,
                )
            )

            # ------------------------------------------------
            # Search CURRENT markdown.
            # ------------------------------------------------

            position = markdown.find(
                table_markdown
            )

            if position == -1:

                raise RuntimeError(
                    f"Could not locate Table "
                    f"{table_number} "
                    f"({docling_ref}) in the "
                    f"exported Markdown.\n"
                    f"This means Docling's table export "
                    f"does not match the document export."
                )

            markdown = (
                markdown[:position]
                + replacement
                + markdown[
                    position
                    + len(table_markdown):
                ]
            )

            replacement_records.append(
                {
                    "table_number": table_number,

                    "element_index": element_index,

                    "page_start": page_start,

                    "page_end": page_end,

                    "docling_ref": docling_ref,

                    "table_file": table_path,

                    "provenance": table.get(
                        "provenance",
                        [],
                    ),

                    "summary": summary,
                }
            )

            self.stats[
                "tables_processed"
            ] += 1

            print(
                f"   ✓ Table {table_number} "
                f"processed."
            )

        return (
            markdown,
            replacement_records,
        )

    # ========================================================
    # PROCESS ONE DOCUMENT
    # ========================================================

    def process_document(
        self,
        file_path: Path,
    ) -> bool:

        document_id = (
            self.create_document_id(
                file_path
            )
        )

        print("\n" + "=" * 80)
        print(
            f"PROCESSING: {file_path.name}"
        )
        print("=" * 80)

        try:

            # ------------------------------------------------
            # 1. DOCLING CONVERSION
            # ------------------------------------------------

            print(
                "Converting document "
                "with Docling..."
            )

            result = (
                self.converter.convert(
                    str(file_path)
                )
            )

            document = result.document

            if hasattr(result, "input") and hasattr(result.input, "_backend"):
                result.input._backend.unload()

            print(
                "     ✓ Conversion complete."
            )

            # ------------------------------------------------
            # 2. HEADING HIERARCHY POSTPROCESSING
            # ------------------------------------------------
            #
            # IMPORTANT:
            #
            # docling-hierarchical-pdf works on the
            # ConversionResult, not on exported Markdown.
            #
            # It modifies result.document in place.
            #
            # Therefore this MUST happen before:
            #
            #   - element extraction
            #   - Markdown export
            #
            # ------------------------------------------------

            print(
                "2/7  Correcting heading hierarchy "
                "with docling-hierarchical-pdf..."
            )

            self.apply_heading_hierarchy(
                result=result,
                source_path=file_path,
            )

            document = result.document

            # ------------------------------------------------
            # 3. EXTRACT STRUCTURED ELEMENTS
            # ------------------------------------------------

            print(
                "3/7  Extracting elements "
                "and PDF provenance..."
            )

            elements = (
                self.extract_elements(
                    document
                )
            )

            print(
                f"     ✓ {len(elements)} "
                f"elements captured."
            )

            # ------------------------------------------------
            # 4. EXPORT COMPLETE MARKDOWN
            # ------------------------------------------------

            print(
                "4/7  Exporting Markdown..."
            )

            original_markdown = (
                document.export_to_markdown(
                    escape_html=True,
                    escape_underscores=True,
                    enable_chart_tables=True,
                    compact_tables=False,
                )
            )

            if not original_markdown.strip():

                raise RuntimeError(
                    "Docling produced empty Markdown."
                )

            print(
                "     ✓ Markdown exported."
            )

            # ------------------------------------------------
            # 5. EXTRACT TABLES DIRECTLY FROM DOCLING
            # ------------------------------------------------

            print(
                "5/7  Extracting tables..."
            )

            table_records = (
                self.extract_table_records(
                    document=document,
                    elements=elements,
                )
            )

            print(
                f"     ✓ "
                f"{len(table_records)} "
                f"table(s) found."
            )

            # ------------------------------------------------
            # 6. SAVE TABLES + GENERATE SUMMARIES
            # ------------------------------------------------

            replacement_records: List[
                Dict[str, Any]
            ] = []

            if table_records:

                print(
                    "6/7  Processing tables..."
                )

                (
                    final_markdown,
                    replacement_records,
                ) = self.replace_tables(
                    original_markdown=original_markdown,
                    table_records=table_records,
                    document_id=document_id,
                    document_name=file_path.name,
                    output_dir=OUTPUT_DIR,
                )

            else:

                print(
                    "6/7  No tables to process."
                )

                final_markdown = (
                    original_markdown
                )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if table_records:

                if len(replacement_records) != len(
                    table_records
                ):

                    raise RuntimeError(
                        "Table processing count mismatch. "
                        f"Found {len(table_records)} "
                        f"tables but processed "
                        f"{len(replacement_records)}."
                    )

            # ------------------------------------------------
            # 7. SAVE OUTPUTS
            # ------------------------------------------------

            print(
                "7/7  Saving outputs..."
            )

            documents_dir = (
                OUTPUT_DIR / "documents"
            )

            manifests_dir = (
                OUTPUT_DIR / "manifests"
            )

            documents_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            manifests_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ------------------------------------------------
            # Main Markdown
            # ------------------------------------------------

            markdown_path = (
                documents_dir
                / f"{document_id}.md"
            )

            markdown_path.write_text(
                final_markdown,
                encoding="utf-8",
            )

            # ------------------------------------------------
            # Manifest
            # ------------------------------------------------

            manifest = {

                "schema_version": "1.1",

                "stage": (
                    "stage_1_docling_ingestion"
                ),

                "document_id": document_id,

                "document_name": (
                    file_path.name
                ),

                "source_file": (
                    str(file_path)
                ),

                "processed_at": (
                    datetime.now().isoformat()
                ),

                "docling_document_pages": (
                    get_document_page_count(document)
                ),

                "markdown_file": (
                    f"documents/"
                    f"{document_id}.md"
                ),

                # ------------------------------------------------
                # SOURCE ELEMENTS
                # ------------------------------------------------
                #
                # These elements are extracted AFTER
                # docling-hierarchical-pdf processing.
                #
                # Therefore hierarchy_level represents the
                # corrected hierarchy.
                # ------------------------------------------------

                "elements": elements,

                # ------------------------------------------------
                # TABLES
                # ------------------------------------------------

                "tables": replacement_records,
            }

            manifest_path = (
                manifests_dir
                / f"{document_id}.json"
            )

            manifest_path.write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print(
                f"     ✓ Markdown: "
                f"{markdown_path}"
            )

            print(
                f"     ✓ Manifest: "
                f"{manifest_path}"
            )

            print(
                "     ✓ Document completed."
            )

            self.stats[
                "files_processed"
            ] += 1

            return True

        except Exception as exc:

            self.stats[
                "files_failed"
            ] += 1

            self.stats[
                "errors"
            ].append(
                {
                    "file": file_path.name,
                    "error": str(exc),
                }
            )

            print(
                f"\n     ❌ FAILED: "
                f"{file_path.name}"
            )

            print(
                f"     Error: {exc}"
            )

            return False

    # ========================================================
    # PROCESS ALL DOCUMENTS
    # ========================================================

    def run(
        self,
        input_dir: Path = INPUT_DIR,
    ) -> None:

        files = (
            self.find_input_files(
                input_dir
            )
        )

        self.stats[
            "files_found"
        ] = len(files)

        if not files:

            print(
                f"No supported documents "
                f"found in {input_dir}"
            )

            return

        print("\n" + "=" * 80)
        print(
            "STAGE 1 - DOCLING INGESTION"
        )
        print("=" * 80)

        print(
            f"Input : {input_dir}"
        )

        print(
            f"Output: {OUTPUT_DIR}"
        )

        print(
            f"Files : {len(files)}"
        )

        print(
            f"LLM   : Groq / {self.model}"
        )

        print(
            "Hierarchy: docling-hierarchical-pdf"
        )

        print("=" * 80)

        for index, file_path in enumerate(
            files,
            start=1,
        ):

            print(
                f"\n[{index}/{len(files)}]"
            )

            self.process_document(
                file_path
            )

            time.sleep(0.5)

        # ----------------------------------------------------
        # Save statistics
        # ----------------------------------------------------

        stats_path = (
            OUTPUT_DIR
            / "processing_statistics.json"
        )

        stats_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        stats_path.write_text(
            json.dumps(
                self.stats,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # Final report
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print(
            "STAGE 1 COMPLETE"
        )
        print("=" * 80)

        print(
            f"Files found       : "
            f"{self.stats['files_found']}"
        )

        print(
            f"Files processed   : "
            f"{self.stats['files_processed']}"
        )

        print(
            f"Files failed      : "
            f"{self.stats['files_failed']}"
        )

        print(
            f"Tables found      : "
            f"{self.stats['tables_found']}"
        )

        print(
            f"Tables processed  : "
            f"{self.stats['tables_processed']}"
        )

        print(
            f"Groq LLM calls    : "
            f"{self.stats['llm_calls']}"
        )

        print(
            f"Tokens used       : "
            f"{self.stats['total_tokens']:,}"
        )

        print(
            f"Errors            : "
            f"{len(self.stats['errors'])}"
        )

        print(
            f"\nStatistics saved to:"
            f"\n{stats_path}"
        )

        print("=" * 80)


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    processor = DoclingIngestion(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
    )

    processor.run(
        input_dir=INPUT_DIR
    )


if __name__ == "__main__":
    main()