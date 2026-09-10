# RetrievalFoundry — Document QA Application

RetrievalFoundry Document QA is a chat application built on top of the RetrievalFoundry RAG engine. It provides a user-facing interface for asking questions across selected documents and receiving document-grounded answers with citations. This app also features regeneration capability. 

The application is composed of:

- A Streamlit frontend for document selection and chat interaction
- A FastAPI backend for chat orchestration
- The RetrievalFoundry RAG engine for retrieval and LLM response generation
- MongoDB for conversation and response persistence

# Architecture

The chat flow is:

```text
                         ┌─────────────────────────────┐
                         │            User             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Streamlit Frontend      │
                         │  Document Selection + Chat   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │       FastAPI Backend        │
                         │     Chat Orchestration       │
                         └──────────────┬──────────────┘
                                        │
                            ▼
                         ┌─────────────────────────────┐
                         │   Load Conversation History  │
                         │          from MongoDB        │
                         └──────────────┬──────────────┘
                            │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                 RAG Engine /retrieve         RAG Engine /chat
                                 + History Context
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                          Summary Response + Citation  │
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │  MongoDB Conversation Store │
                         └─────────────────────────────┘
```

## Batch Chat Flow

For questions asked across multiple files, the backend uses an adaptive batch pipeline:

```text
          ┌─────────────────────────────┐
          │       Selected Documents    │
          │          Multiple Files     │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │       FastAPI Backend        │
          │     Chat Orchestration       │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │   Load Conversation History  │
          │          from MongoDB        │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │   Retrieve Chunks per File   │
          │       through /retrieve     │
          └──────────────┬──────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │       Split into Batches     │
          │       Based on Batch Size    │
          └──────────────┬──────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
        Batch Summary 1  Batch Summary 2  Batch Summary N
          │              │              │
          └──────────────┼──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │   Final Summary Generation   │
          │ through /chat + history      │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │  Citation Mapping           │
          │                             │
          └──────────────┬──────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │  Persist Response + Return   │
          │       Answer and Citations   │
          └─────────────────────────────┘
```

## Regeneration Flow

The regeneration flow is supported through a separate endpoint and can reuse the original question or accept an edited question.

```text
                         ┌─────────────────────────────┐
                         │   Regeneration Request      │
                         │ conversation_id             │
                         │ question_id                 │
                         │ optional edited_question    │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Fetch Original Question and  │
                         │ Response ID from Storage     │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Load Stored History Context  │
                         │       from Response Record   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Archive Previous Response    │
                         │ and Select Question          │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │      Usual Chat Pipeline     │
                         │ Retrieve, Summarize, Cite,   │
                         │ Persist, and Return          │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Update Existing Response by  │
                         │           Response ID        │
                         └─────────────────────────────┘
```

# Application Components

## Streamlit Frontend

The frontend provides:

- Selection of documents from the document library
- A chat interface for submitting questions
- Display of generated answers and source citations
- Answer regeneration with the original or an edited question

The frontend entry point is `fe/app.py`.

## FastAPI Backend

The backend coordinates the application workflow by:

- Creating and continuing conversations
- Calling the RAG engine for chunk retrieval
- Calling the RAG engine for answer generation
- Selecting single-pass or batch summarization depending on number of selected documents
- Building citations from retrieved chunks
- Persisting conversations and responses
- Supporting response regeneration

The backend entry point is `be/app/main.py`.

# Chat Pipeline

The normal chat pipeline is:

1. The user selects one or more documents.
2. The frontend sends the question and selected file IDs to the backend.
3. The backend creates or continues a conversation.
4. Relevant chunks are retrieved from the RAG engine for each selected document.
5. The backend generates a summary response using the retrieved chunks.
6. Source references are mapped back to the retrieved chunks.
7. The response is saved with its conversation and citation metadata.
8. The answer and citations are returned to the frontend.

The summarization strategy is adaptive:

- **Single-pass summarization** is used for smaller document selections.
- **Batch summarization** is used for larger selections, followed by a final combined summary.

# Regeneration

The application supports regeneration of a previously generated answer.

- The original question can be regenerated without modification.
- An edited question can be supplied for a revised answer.
- The previous response is archived.
- Retrieval and summarization are executed again.
- The stored response is updated with the new answer and citations.

> **Storage note:** To reduce storage usage in MongoDB Atlas, archived responses are kept temporarily in a local JSON file. The original conversation and response thread remain stored in MongoDB.

# API

Protected endpoints require the `X-API-Key` request header.

## Chat Endpoints

### `POST /api/v1/chat/ask`

Executes the complete chat pipeline.

Request fields include:

- `question`
- `file_ids`
- `conversation_id`, when continuing an existing conversation
- `created_by`

The response contains the generated summary, citations, conversation ID, question ID, selected file IDs, and token-cost information.

### `POST /api/v1/chat/regenerate`

Regenerates a previous response.

Request fields include:

- `conversation_id`
- `question_id`
- `edited_question`, optionally
- `created_by`

## Conversation Endpoints

- `GET /api/v1/conversations/`
- `GET /api/v1/conversations/{conversation_id}`

These endpoints expose conversation metadata and stored conversation responses.

## Pipeline Endpoints

Individual pipeline stages are also available through:

- `POST /api/v1/pipeline/record-prompt`
- `POST /api/v1/pipeline/retrieve`
- `POST /api/v1/pipeline/summarize`

These provide access to the decomposed pipeline stages in addition to the complete chat endpoints.

## Health Endpoints

- `GET /health`
- `GET /`

# Request and Response Format

The API uses a common response wrapper:

```json
{
  "status": true,
  "item": {},
  "message": "..."
}
```

A successful chat response contains:

- `question_id`
- `conversation_id`
- `summary`
- `citations`
- `token_cost`
- `file_ids`

Each citation contains:

- `file_name`
- `file_id`
- `page_number`
- `excerpt`
- `section_path`

# RAG Engine Integration

The application communicates with the separate RAG engine through its HTTP API.

## Retrieval

The backend calls `POST /retrieve` to obtain relevant chunks for a selected document. Retrieved chunks include the content and metadata required for summarization and citation generation.

## LLM Generation

The backend calls `POST /chat` to generate responses from retrieved context. The request can include prompts and conversation history.

The RAG engine base URL is configured through `RAG_API_BASE_URL`.

# Setup

The frontend and backend use separate Python environments.

## Backend

From the `rag-app/rag-app` directory:

```powershell
cd be
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

Configure the backend `.env` file with the required API, RAG engine, MongoDB, and storage settings.

## Frontend

In a separate terminal, from the `rag-app/rag-app` directory:

```powershell
cd fe
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

# Repository Structure

```text
rag-app/
│
├── README.md
│
├── be/
│   ├── app/
│   │   ├── adapters/       # RAG engine and provider adapters
│   │   ├── interfaces/     # Provider interfaces
│   │   ├── prompts/        # Prompt templates
│   │   ├── repositories/   # Conversation and response persistence
│   │   ├── routes/         # FastAPI routes
│   │   ├── schemas/        # Request, response, and state models
│   │   ├── services/       # Pipeline, retrieval, summary, and citation logic
│   │   └── utils/          # Shared utilities
│   ├── data/               # Local application data
│   ├── tests/              # Backend tests
│   ├── requirements.txt
│   └── .env.example
│
└── fe/
    ├── app.py              # Streamlit application
    └── requirements.txt
```

