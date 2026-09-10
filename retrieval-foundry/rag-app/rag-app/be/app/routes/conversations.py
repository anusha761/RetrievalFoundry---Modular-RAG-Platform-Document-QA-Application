"""Conversation routes — list and view conversation history."""
import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.dependencies import verify_api_key
from app.schemas.responses import ResponseSchema, ConversationItem
from app.repositories import conversation_repo, response_repo

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"], dependencies=[Depends(verify_api_key)])


@router.get("/", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def list_conversations():
    """List all active conversations."""
    try:
        logger.info("Running API : conversations/list")

        conversations = await conversation_repo.list_conversations()
        items = [
            ConversationItem(
                conversation_id=c.conversation_id,
                last_question=c.last_question,
                last_question_id=c.last_question_id,
                file_ids=c.file_ids,
                created_date=c.created_date,
                status=c.status
            ).model_dump()
            for c in conversations
        ]

        logger.info(f"conversations/list complete: {len(items)} conversations found")
        response = ResponseSchema(status=True, item=items, message=f"Found {len(items)} conversations")
        return response

    except Exception as e:
        logger.error(f"Error conversations/list - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{conversation_id}", response_model=ResponseSchema, status_code=status.HTTP_200_OK)
async def get_conversation_history(conversation_id: str):
    """Get a conversation with its full Q&A history."""
    try:
        logger.info(f"Running API : conversations/{conversation_id}")

        conversation = await conversation_repo.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get all responses for this conversation
        history = await response_repo.get_conversation_history(conversation_id, limit=50)
        history_items = [
            {
                "response_id": r.response_id,
                "question_id": r.question_id,
                "prompt_question": r.prompt_question,
                "summary": r.summary,
                "citations": r.citations,
                "token_cost": r.token_cost,
                "is_regeneration": r.is_regeneration,
                "created_date": r.created_date
            }
            for r in history
        ]

        result = {
            "conversation": conversation.model_dump(),
            "history": history_items
        }

        logger.info(f"conversations/{conversation_id} complete: {len(history_items)} turns")
        response = ResponseSchema(status=True, item=result, message="Conversation retrieved successfully")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error conversations/{conversation_id} - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e
