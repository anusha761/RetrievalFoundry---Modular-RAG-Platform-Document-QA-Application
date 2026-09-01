# RetrievalFoundry — Modular RAG Engine

RetrievalFoundry is a modular RAG engine for document-grounded question answering, designed around structure-aware ingestion, provenance, hybrid retrieval, reranking, table preservation, independently consumable APIs, and LLM-based evaluation.

## Engineering Focus

RetrievalFoundry goes beyond a vector-only RAG pipeline through:

- **Structure-aware ingestion:** Docling-based document structure and provenance
- **Hierarchical chunking:** section-aware chunk boundaries with metadata propagation
- **Hybrid retrieval:** dense embeddings + BM25
- **Multi-stage ranking:** RRF fusion followed by cross-encoder reranking
- **Table preservation:** original tables stored separately and resolved at retrieval time
- **Modular serving:** retrieval and generation exposed independently through FastAPI
- **Evaluation:** DeepEval with an independent LLM-as-a-judge
- **Idempotent ingestion:** deterministic Qdrant point IDs derived from document/chunk identity


Why Qdrant used as vectorDB?

Qdrant was chosen for its support for payload-based metadata filtering, which allows retrieval to be constrained to specific documents selected by the application. It also supports dense and sparse vectors within the same collection, making it well suited to theproject's hybrid retrieval architecture.


# Architecture

```text

                         ┌─────────────────────────────┐
                         │        PDF Documents        │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │    Document Ingestion       │
                         │          Docling            │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                         ▼              ▼              ▼
                    Markdown      Provenance       Original
                    + Structure     Manifest        Tables
                         │              │              │
                         └──────────────┼──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │   Chunking + Metadata       │
                         │  Section-aware processing   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │     Dense + Sparse          │
                         │       Embeddings            │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │           Qdrant            │
                         │      Hybrid Retrieval       │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                   Dense Search                  Sparse BM25
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                              RRF Candidate Fusion
                                        │
                                        ▼
                              Cross-Encoder Reranker
                                        │
                                        ▼
                                  Top-K Chunks
                                        │
                                        ▼
                                Table Resolution
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                 Retrieval API                  Application LLM API
                                                   (Groq)


```

# RAG Pipeline
## 1. Document Ingestion

PDF documents are processed using Docling (docling-hierarchical-pdf) rather than treating the PDF as an unstructured text blob.

The ingestion stage produces:

- Markdown representation of each document
- Document-level provenance manifest
- Original PDF page information
- Structural element information
- Extracted tables
- Retrieval-oriented table summaries

The pipeline uses document heading and section information to support downstream structure-aware chunking.

### Table Handling

Tables receive dedicated handling rather than being discarded or blindly flattened into surrounding text.

```text

                 Original PDF Table
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Complete Table         Retrieval Summary
       stored separately       in main Markdown
              │                     │
              └──────────┬──────────┘
                         ▼
                  Table Reference
                         │
                         ▼
                  Retrieved Chunk
                         │
                         ▼
                   Table Resolver
                         │
                         ▼
              Original Table Content

```

The complete original table is preserved separately, while the main Markdown contains a retrieval-oriented representation.

When a retrieved chunk contains a table reference, the retrieval layer can resolve the original table and provide it alongside the retrieved chunk.


## 2. Structure-Aware Chunking and Metadata

The chunking stage operates on the Markdown representation produced during ingestion.

Rather than splitting documents using only a fixed character boundary, the pipeline uses available heading and section structure to preserve logical document context where possible.

The chunking pipeline:

- Preserves section hierarchy
- Keeps logical sections together where possible
- Descends into child sections when a section is too large
- Handles heading-only sections without silently discarding their semantic information
- Enforces a maximum token limit
- Preserves table references
- Attaches document and page metadata to chunks

Each chunk can contain metadata such as:

- document_id
- chunk_id
- page_no
- section_path
- table_ref
- chunk_text
- retrieval_text

This metadata is retained throughout the retrieval pipeline.The provenance manifest is treated as the authoritative source for original PDF page numbers rather than attempting to infer page numbers later from the generated Markdown.

## 3. Retrieval

### Retrieval Architecture

The retrieval pipeline uses a multi-stage approach:

```text
                         User Query
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Dense Retrieval            Sparse Retrieval
          Semantic Search                BM25
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    Reciprocal Rank
                       Fusion (RRF)
                              │
                              ▼
                     Candidate Chunks
                              │
                              ▼
                   Cross-Encoder Reranker
                              │
                              ▼
                        Top-K Chunks
                              │
                              ▼
                      Table Resolution
                              │
                              ▼
                    Final Retrieval Context
```



### Retrieval

The retrieval pipeline is designed to balance **candidate recall** and **ranking precision** across multiple stages:

- **Dense retrieval:** Semantic matching for conceptually relevant content.
- **Sparse retrieval:** Complements with Lexical matching for exact terms, identifiers, and domain-specific vocabulary.
- **RRF fusion:** Combines complementary retrieval signals to improve candidate coverage.
- **Cross-encoder reranking:** Improves ranking precision by performing deeper query–chunk relevance scoring.
- **Table resolution:** Resolves referenced tables before constructing the final retrieval context.


## 4. Modular API Layer

The RAG engine exposes its functionality through FastAPI. The API layer is intentionally separated from the underlying retrieval and generation services.

### Retrieval API

The retrieval endpoint exposes the complete retrieval pipeline:

```text
Query
  ↓
Dense + Sparse Retrieval
  ↓
RRF
  ↓
Cross-Encoder Reranking
  ↓
Table Resolution
  ↓
Ranked Retrieval Context
```

The retrieval functionality can therefore be consumed independently by other applications.

### LLM Chat API

The LLM chat functionality is implemented separately from retrieval.

```text
Client
  ↓
FastAPI
  ↓
Chat Service
  ↓
Groq
  ↓
LLM Response
```

### Qdrant Deployment Options

The repository contains separate ingestion paths for local development and Qdrant Cloud.

```text
qdrant_ingestion_local.py
        │
        ▼
   Local Qdrant

qdrant_ingestion.py
        │
        ▼
   Qdrant Cloud
```

## 5. Evaluation

RAG evaluation using deepeval is maintained as a separate component from the core retrieval implementation.

The evaluation pipeline uses DeepEval with Gemini as an independent LLM-as-a-judge.

```text
                 RAG Output
                     │
                     ▼
              DeepEval Pipeline
                     │
                     ▼
             Gemini LLM-as-Judge
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Answer     Contextual   Faithfulness
      Relevancy   Relevancy

The application and evaluation layers intentionally use different models:

Application
    ↓
Groq
    ↓
Answer Generation


Evaluation
    ↓
Gemini
    ↓
LLM-as-a-Judge
```

### Evaluation metrics:

- Answer Relevancy
- Contextual Relevancy
- Faithfulness


## 6. Repository Structure

```text
rag-engine/
│
├── README.md
├── requirements.txt
├── requirements-evaluation.txt
├── .env.example
├── .gitignore
│
├── src/
│   └── rag_engine/
│       │
│       ├── ingestion/
│       │   ├── document_ingestion.py # step 1 (pdf-markdown)
│       │   ├── chunking_metadata.py # step 2 (chunking with metadata)
│       │   ├── qdrant_ingestion_local.py # step 3
│       │   └── qdrant_ingestion.py # step 3
│       │
│       ├── retrieval/
│       │   ├── retriever.py
│       │   ├── reranker.py
│       │   ├── table_resolver.py
│       │   └── retrieval_service.py
│       │
│       ├── generation/
│       │   └── chat_service.py
│       │
│       └── api/
│           └── app.py
│
├── evaluation/
│   └── deepeval/
│       ├── rag_evaluation.py
│       └── deepeval_input.json
│
└── data/
    ├── input/
    ├── stage_1/
    └── stage_2/
```

## Setup

1. Create the environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```
2. Install dependencies

```bash
pip install -r requirements.txt
```

Evaluation dependencies are maintained separately:

```bash
pip install -r requirements-evaluation.txt
```
3. Configure environment variables

Create a .env file based on .env.example.

Required credentials depend on the execution path being used.

## Running the Pipeline

### Step 1 — Document Ingestion

Place input PDFs under:

```
data/input/
```

Run:

```bash
python src/rag_engine/ingestion/document_ingestion.py
```

This generates the Stage 1 document representations and provenance artifacts.

### Step 2 — Chunking and Metadata Generation

Run:

```bash
python src/rag_engine/ingestion/chunking_metadata.py
```

This generates the retrieval-ready chunk data.

### Step 3 — Qdrant Ingestion

For local Qdrant:

```bash
python src/rag_engine/ingestion/qdrant_ingestion_local.py
```

For Qdrant Cloud:

```bash
python src/rag_engine/ingestion/qdrant_ingestion.py
```
## Running the FastAPI Service

Start the API using Uvicorn:

```bash
uvicorn rag_engine.api.app:app --reload
```

The FastAPI application exposes the retrieval and LLM functionality through HTTP endpoints.

Interactive API documentation is available through FastAPI's Swagger UI.

## Running Evaluation

The evaluation pipeline accepts a JSON dataset containing:

- user queries
- retrieved chunks
- generated answers

Example:

```bash
python evaluation/deepeval/rag_evaluation.py \
    --input evaluation/deepeval/deepeval_input.json \
    --output evaluation/deepeval/deepeval_results.xlsx
```

## Evaluation Results

The resulting evaluation report contains scores for:

- Answer Relevancy
- Contextual Relevancy
- Faithfulness

The RAG pipeline was evaluated across **10 questions** using [DeepEval](https://deepeval.com/).

#### Overall Scores

| Metric | Average Score |
|---|---:|
| Answer Relevancy | **0.985** |
| Contextual Relevancy | **0.960** |
| Faithfulness | **0.990** |

For the complete question-level evaluation results, including the generated answers and individual metric scores, see the [**full evaluation results (Excel)**](https://github.com/anusha761/RetrievalFoundry---Modular-RAG-Platform-Document-QA-Application/blob/main/retrieval-foundry/rag-engine/evaluation/deepeval/RAG_Evaluation_Results.xlsx).


