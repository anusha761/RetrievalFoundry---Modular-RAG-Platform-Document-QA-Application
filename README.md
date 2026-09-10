# RetrievalFoundry

## An End-to-End Document Intelligence Platform

RetrievalFoundry is a modular, document-grounded question-answering platform for PDF documents. It combines structure-aware ingestion, hybrid retrieval, reranking, table preservation, provenance tracking, and independent evaluation with a multi-document chat application that supports citations, conversation history, and answer regeneration.

The repository contains two cooperating systems:

- **[RAG Engine](retrieval-foundry/rag-engine/README.md):** ingestion, indexing, retrieval, reranking, table resolution, generation APIs, and evaluation.
- **[Document QA Application](retrieval-foundry/rag-app/rag-app/README.md):** Streamlit chat experience and FastAPI orchestration for multi-document conversations, citations, persistence, and regeneration.

## Demo Video 

[View the demo](https://drive.google.com/file/d/1iQsjGUUFfN20IjyZnHPhNwZpCf1MlMsF/view?usp=drive_link)

## Why This Project

RetrievalFoundry goes beyond a basic vector-search chatbot by treating document structure, provenance, tables, retrieval quality, and evaluation as first-class concerns.

- **Structure-aware ingestion:** PDFs are processed with Docling (docling-hierarchical-pdf) to preserve headings, sections, page information, and document structure.
- **Hierarchical chunking:** Chunks follow document sections where possible and retain document, page, section, and table metadata.
- **Hybrid retrieval:** Dense semantic search is combined with sparse BM25 retrieval for both conceptual queries and exact financial terminology.
- **Multi-stage ranking:** Reciprocal Rank Fusion combines retrieval signals before cross-encoder reranking improves candidate precision.
- **Table-aware retrieval:** Original tables are preserved separately and resolved when referenced by retrieved chunks.
- **Cited answers:** Source references are mapped back to document names, pages, sections, and excerpts.
- **Multi-document chat:** Users can ask questions across selected documents, with adaptive single-pass or batch summarization.
- **Conversation-aware generation:** Conversation history is carried into answer generation and regeneration.
- **Regeneration:** Existing answers can be regenerated with the original question or an edited question while retaining the response thread context.
- **Independent evaluation:** DeepEval measures answer relevancy, contextual relevancy, and faithfulness using an independent LLM judge.

## System Architecture

```text
                              PDF Documents
                                    |
                                    v
                 +-----------------------------------------+
                 | Structure-Aware Document Ingestion     |
                 | Docling, Markdown, Provenance, Tables  |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | Hierarchical Chunking + Metadata       |
                 | Sections, Pages, Tables, Chunk Identity |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | Dense + Sparse Indexing                |
                 | Embeddings, BM25, Qdrant              |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | Hybrid Retrieval                       |
                 | Dense Search + BM25 + RRF              |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | Cross-Encoder Reranking + Table Resolve |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | RAG Engine FastAPI APIs                 |
                 | /retrieve and /chat                    |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | FastAPI Document QA Application        |
                 | Retrieval, Summarization, Citations,   |
                 | History, Persistence (MongoDB),        |
                 | Regeneration                           |
                 +-------------------+---------------------+
                                     |
                                     v
                 +-----------------------------------------+
                 | Streamlit Document QA Interface        |
                 +-----------------------------------------+
```

## Retrieval and Answer Generation

The retrieval path is designed to balance recall and precision:

```text
User Query
    |
    +--> Dense Retrieval --------+
    |                            |
    +--> Sparse BM25 ------------+--> RRF Fusion
                                      |
                                      v
                              Candidate Chunks
                                      |
                                      v
                            Cross-Encoder Reranking
                                      |
                                      v
                                Top-K Chunks
                                      |
                                      v
                               Table Resolution
                                      |
                                      v
                           Cited Retrieval Context
```

The RAG engine exposes retrieval and generation independently. This allows the application layer to orchestrate retrieval, conversation context, summarization, citation mapping, and persistence without coupling those responsibilities to the indexing implementation.

## Application Workflow

The Document QA application provides the user-facing experience:

1. The user selects one or more documents.
2. The application retrieves relevant chunks for each selected file.
3. Smaller selections use single-pass summarization.
4. Larger selections are split into batches, summarized concurrently, and combined into a final answer.
5. Generated source references are mapped back to retrieved chunks.
6. The answer, citations, token usage, and conversation metadata are persisted.
7. The application returns a document-grounded answer with source excerpts.
8. Existing answers can be regenerated with the original or an edited question while reusing stored conversation context.

## Table and Provenance Handling

Tables are not flattened and discarded as ordinary text. During ingestion:

- Original tables are stored separately.
- Retrieval-oriented table summaries are included in the document representation.
- Table references are preserved in chunk metadata.
- Retrieved table references can be resolved back to the original table content.

Provenance manifests remain the authoritative source for original PDF page numbers and sections. This allows citations to retain document and page-level traceability through ingestion, retrieval, generation, and response presentation.

## Modular APIs

### RAG Engine

- `POST /retrieve` executes hybrid retrieval, RRF fusion, reranking, and table resolution.
- `POST /chat` generates an LLM response from prompts, retrieved context, and optional conversation history.
- `GET /health` reports service health.

### Document QA Application

- `POST /api/v1/chat/ask` executes the complete chat pipeline.
- `POST /api/v1/chat/regenerate` regenerates a previous answer.
- `GET /api/v1/conversations/` exposes conversation metadata.
- `GET /api/v1/conversations/{conversation_id}` returns a conversation thread.
- Decomposed pipeline endpoints are available for prompt recording, retrieval, and summarization stages.

## Evaluation

Evaluation is maintained separately from application generation so that answer quality is assessed by an independent model:

```text
                         RAG Output
                              |
                              v
                         DeepEval
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       Answer Relevancy  Contextual       Faithfulness
                         Relevancy
                              
```

The application uses Groq for answer generation, while Gemini is used independently as the evaluation judge. This separation helps evaluate the generated answers without relying on the same model for both generation and assessment.

## Technology Stack

| Area | Technology |
| --- | --- |
| Language | Python |
| API framework | FastAPI |
| User interface | Streamlit |
| Document processing | Docling |
| Vector database | Qdrant |
| Sparse retrieval | BM25 |
| Reranking | Cross-encoder |
| Generation API | Groq |
| Answer Generation | GPT-OSS 120B |
| Application persistence | MongoDB |
| Evaluation | DeepEval + Gemini |

## Repository Structure

```text
retrieval-foundry/
|
+-- README.md                         # Project overview
|
+-- rag-engine/
|   +-- src/rag_engine/               # Ingestion, retrieval, generation, APIs
|   +-- evaluation/                   # DeepEval evaluation pipeline
|   +-- data/                         # Input and generated pipeline artifacts
|   +-- README.md                     # RAG engine documentation
|
+-- rag-app/
    +-- rag-app/
        +-- be/                       # FastAPI application backend
        +-- fe/                       # Streamlit frontend
        +-- README.md                 # Document QA application documentation
```

## Future Scope
- Multi-document latency optimization through caching and adaptive reranking
- Observability for token usage, latency, and inference cost.
- Expansion of automated evaluation and regression benchmarking for retrieval and answer quality.
