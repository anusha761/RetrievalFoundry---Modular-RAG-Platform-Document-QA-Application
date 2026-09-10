"""Chat routes — ask and regenerate endpoints."""
import time
import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import verify_api_key, get_llm, get_vector_store
from app.interfaces.llm_interface import LLMInterface
from app.interfaces.vector_interface import VectorStoreInterface
from app.schemas.requests import ChatRequest, RegenerateRequest
from app.schemas.responses import ResponseSchema
from app.services import pipeline_service
from app.config import load_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"], dependencies=[Depends(verify_api_key)])


@router.post("/ask", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def ask(
    request_body: ChatRequest,
    llm: LLMInterface = Depends(get_llm),
    vector_store: VectorStoreInterface = Depends(get_vector_store)
):
    """Monolithic endpoint: question in, full answer out.

    Orchestrates the entire RAG pipeline in one call.
    """
    try:
        logger.info("Running API : chat/ask")
        raw_request = request_body.model_dump()
        logger.info(f"ChatRequest request body: {raw_request}")

        settings = load_settings()

        # Validate file count
        if len(request_body.file_ids) > settings.MAX_FILES_ALLOWED:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {settings.MAX_FILES_ALLOWED} files allowed. Got {len(request_body.file_ids)}."
            )

        st = time.time()
        result = await pipeline_service.run_full_pipeline(
            request=request_body,
            llm=llm,
            vector_store=vector_store
        )
        et = time.time()
        logger.info(f"Time taken for chat/ask pipeline: {et - st:.2f} seconds")

        logger.info(f"chat/ask complete: {result.question_id}")
        response = ResponseSchema(status=True, item=result.model_dump(), message="Chat response generated successfully")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error chat/ask - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/regenerate", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def regenerate(
    request_body: RegenerateRequest,
    llm: LLMInterface = Depends(get_llm),
    vector_store: VectorStoreInterface = Depends(get_vector_store)
):
    """Regenerate a previous answer with a fresh LLM call."""
    try:
        logger.info("Running API : chat/regenerate")
        raw_request = request_body.model_dump()
        logger.info(f"RegenerateRequest request body: {raw_request}")

        st = time.time()
        result = await pipeline_service.regenerate_response(
            conversation_id=request_body.conversation_id,
            question_id=request_body.question_id,
            edited_question=request_body.edited_question,
            llm=llm,
            vector_store=vector_store
        )
        et = time.time()
        logger.info(f"Time taken for chat/regenerate: {et - st:.2f} seconds")

        response = ResponseSchema(status=True, item=result.model_dump(), message="Regenerated successfully")
        return response

    except ValueError as e:
        logger.error(f"Error chat/regenerate - {str(e)}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error chat/regenerate - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
