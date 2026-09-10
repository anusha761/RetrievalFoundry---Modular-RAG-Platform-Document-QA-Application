"""Real LLM adapter that calls the RAG API's /chat endpoint."""
import time
import httpx
import structlog
from typing import Tuple
from app.interfaces.llm_interface import LLMInterface
from app.config import load_settings

logger = structlog.get_logger()


class RagApiLLM(LLMInterface):
    """Calls the external RAG API /chat endpoint for LLM inference."""

    def __init__(self, base_url: str = None):
        settings = load_settings()
        self.base_url = base_url or settings.RAG_API_BASE_URL

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        conversation_history: list = None
    ) -> Tuple[str, dict]:
        """Send a chat request to the RAG API's /chat endpoint.

        Args:
            system_prompt: System instruction for the LLM.
            user_prompt: User message/query.
            max_tokens: Maximum tokens in response (not used by current API, reserved for future).
            temperature: Sampling temperature (not used by current API, reserved for future).
            conversation_history: Optional list of prior-turn message dicts
                [{"role": "user"/"assistant", "content": "..."}]. Sent as-is to the API.

        Returns:
            Tuple of (response_text, token_usage_dict).
            token_usage_dict defaults to {"prompt_tokens": 0, "completion_tokens": 0} until API adds token tracking.
        """
        url = f"{self.base_url}/chat"

        payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "conversation_history": conversation_history or []
        }

        st = time.time()
        logger.info(f"rag_api_llm.calling - url: {url}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()

            et = time.time()
            logger.info(f"Time taken for rag_api_llm /chat call: {et - st:.2f} seconds")

            content = data.get("response", "")

            # Token usage not yet available from this API — default to 0
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0}

            return content, token_usage

        except httpx.HTTPStatusError as e:
            logger.error(f"rag_api_llm.http_error - status: {e.response.status_code}, detail: {e.response.text}")
            raise RuntimeError(f"RAG API /chat returned {e.response.status_code}: {e.response.text}") from e
        except httpx.ConnectError as e:
            logger.error(f"rag_api_llm.connection_error - {str(e)}")
            raise RuntimeError(f"Cannot connect to RAG API at {url}. Is it running?") from e
        except Exception as e:
            logger.error(f"rag_api_llm.unexpected_error - {str(e)}")
            raise
