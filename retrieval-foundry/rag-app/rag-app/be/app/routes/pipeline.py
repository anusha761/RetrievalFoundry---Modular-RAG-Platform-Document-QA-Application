"""Pipeline routes — broken-out step-by-step endpoints.

Each endpoint follows the original project's pattern:
- Parse request body via model_dump()
- Log entry and exit
- Time the service call
- Return ResponseSchema(status, item, message)
- Catch exceptions with logger.error + HTTPException
"""
import time
import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import verify_api_key, get_llm, get_vector_store
from app.interfaces.llm_interface import LLMInterface
from app.interfaces.vector_interface import VectorStoreInterface
from app.schemas.requests import RetrieveRequest, SummarizeRequest, RecordPromptRequest
from app.schemas.responses import (
    ResponseSchema, RecordPromptResponseItem,
    RetrieveResponseItem, ChatResponseItem, Citation, TokenCost
)
from app.schemas.models import PipelineState
from app.services import pipeline_service, retrieval_service
from app.config import load_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline"], dependencies=[Depends(verify_api_key)])


@router.post("/record-prompt", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def record_prompt(request_body: RecordPromptRequest):
    """Step 1: Record the question and create/continue a conversation."""
    try:
        logger.info("Running pipeline/record-prompt")
        raw_request = request_body.model_dump()
        logger.info(f"RecordPromptRequest request body: {raw_request}")

        state = PipelineState(
            question=request_body.question,
            file_ids=request_body.file_ids,
            created_by=request_body.created_by.model_dump()
        )

        st = time.time()
        state = await pipeline_service.step_record_prompt(state, request_body.conversation_id)
        et = time.time()
        logger.info(f"Time taken for record-prompt: {et - st:.2f} seconds")

        result = RecordPromptResponseItem(
            question_id=state.question_id,
            conversation_id=state.conversation_id
        )

        logger.info(f"record-prompt complete: {result}")
        response = ResponseSchema(status=True, item=result.model_dump(), message="Prompt recorded successfully")
        return response

    except Exception as e:
        logger.error(f"Error pipeline/record-prompt - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e





@router.post("/retrieve", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def retrieve(
    request_body: RetrieveRequest,
    vector_store: VectorStoreInterface = Depends(get_vector_store)
):
    """Step 3: Retrieve relevant chunks from vector store."""
    try:
        logger.info("Running pipeline/retrieve")
        raw_request = request_body.model_dump()
        logger.info(f"RetrieveRequest request body: {raw_request}")

        st = time.time()
        chunks = await retrieval_service.retrieve_chunks(
            vector_store=vector_store,
            retriever_query=request_body.retriever_query,
            file_ids=request_body.file_ids
        )
        et = time.time()
        logger.info(f"Time taken for retrieve: {et - st:.2f} seconds")

        result = RetrieveResponseItem(retrieved_chunks=chunks)
        logger.info(f"retrieve complete: {len(chunks)} files retrieved")
        response = ResponseSchema(status=True, item=result.model_dump(), message="Retrieval complete")
        return response

    except Exception as e:
        logger.error(f"Error pipeline/retrieve - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/summarize", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def summarize(
    request_body: SummarizeRequest,
    llm: LLMInterface = Depends(get_llm)
):
    """Step 4: Generate summary (handles both single-pass and batch mode internally)."""
    try:
        logger.info("Running pipeline/summarize")
        raw_request = request_body.model_dump()
        logger.info(f"SummarizeRequest request body keys: {list(raw_request.keys())}")

        settings = load_settings()

        state = PipelineState(
            question=request_body.question,
            question_id=request_body.question_id,
            conversation_id=request_body.conversation_id,
            file_ids=request_body.file_ids,
            retriever_query=request_body.retriever_query,
            retrieved_chunks=request_body.retrieved_chunks,
            keywords_token_usage=request_body.keywords_token_usage,
            regenerate=request_body.regenerate,
            created_by=request_body.created_by.model_dump()
        )

        st = time.time()
        state = await pipeline_service.step_summarize(state, llm, settings)
        et = time.time()
        logger.info(f"Time taken for summarize: {et - st:.2f} seconds")

        st = time.time()
        state = await pipeline_service.step_persist(state)
        et = time.time()
        logger.info(f"Time taken for persist: {et - st:.2f} seconds")

        result = ChatResponseItem(
            question_id=state.question_id,
            conversation_id=state.conversation_id,
            summary=state.summary,
            citations=[Citation(**c) for c in state.citations],
            token_cost=TokenCost(
                input_tokens=state.total_input_tokens,
                output_tokens=state.total_output_tokens,
                total_cost=state.total_cost
            ),
            file_ids=state.file_ids
        )

        logger.info(f"summarize complete: {state.question_id}")
        response = ResponseSchema(status=True, item=result.model_dump(), message="Summary generated successfully")
        return response

    except Exception as e:
        logger.error(f"Error pipeline/summarize - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
