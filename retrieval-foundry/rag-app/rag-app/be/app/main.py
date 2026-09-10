"""FastAPI application entry point."""
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat, pipeline, conversations
from app.config import load_settings

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Create app
app = FastAPI(
    title="Document RAG Chat API",
    description=(
        "A multi-document RAG chat API with adaptive pipeline processing, "
        "query refinement, batch summarization, citation generation, "
        "and conversation history."
    ),
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(chat.router)
app.include_router(pipeline.router)
app.include_router(conversations.router)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    settings = load_settings()
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "llm_provider": settings.LLM_PROVIDER,
        "vector_store_provider": settings.VECTOR_STORE_PROVIDER
    }


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint."""
    return {"message": "Document RAG Chat API", "docs": "/docs"}
