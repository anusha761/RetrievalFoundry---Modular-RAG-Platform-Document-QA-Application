"""
CHAT SERVICE

Purpose
-------
Send system and user prompts to Groq using an OpenAI-compatible
client interface.

This module is intentionally separate from the retrieval pipeline.

It does NOT perform:

    - Retrieval
    - Qdrant search
    - Reranking
    - Table resolution
    - Prompt construction

It only calls the Groq API and returns the model response.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


# ==========================================================
# CONFIGURATION
# ==========================================================

GROQ_MODEL_NAME = "openai/gpt-oss-20b"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ==========================================================
# RUNTIME RESOURCE
# ==========================================================

client: Optional[OpenAI] = None


# ==========================================================
# INITIALIZATION
# ==========================================================

def initialize_chat_service() -> None:
    """
    Initialize the Groq client once during application startup.

    The API key is read from the GROQ_API_KEY environment
    variable.
    """

    global client

    if client is not None:
        return

    print("Initializing Groq chat client...")

    # ------------------------------------------------------
    # Load .env
    # ------------------------------------------------------

    load_dotenv()

    # ------------------------------------------------------
    # Read API key
    # ------------------------------------------------------

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in the environment. "
            "Make sure it is defined in your .env file."
        )

    # ------------------------------------------------------
    # Create OpenAI-compatible Groq client
    # ------------------------------------------------------

    client = OpenAI(
        api_key=groq_api_key,
        base_url=GROQ_BASE_URL,
    )

    print("Groq chat client initialized.")


# ==========================================================
# RESOURCE VALIDATION
# ==========================================================

def _ensure_initialized() -> None:
    """
    Ensure the Groq client has been initialized.
    """

    if client is None:

        raise RuntimeError(
            "Groq chat client is not initialized. "
            "Call initialize_chat_service() during "
            "application startup."
        )


# ==========================================================
# CHAT FUNCTION
# ==========================================================

def generate_chat_response(
    system_prompt: str,
    user_prompt: str,
) -> str:
    """
    Send system and user prompts to the Groq model.

    Parameters
    ----------
    system_prompt : str
        System-level instruction.

    user_prompt : str
        User message.

    Returns
    -------
    str
        Model-generated response.
    """

    # ------------------------------------------------------
    # Validate system prompt
    # ------------------------------------------------------

    if not isinstance(system_prompt, str):

        raise TypeError(
            "system_prompt must be a string."
        )

    system_prompt = system_prompt.strip()

    if not system_prompt:

        raise ValueError(
            "system_prompt cannot be empty."
        )

    # ------------------------------------------------------
    # Validate user prompt
    # ------------------------------------------------------

    if not isinstance(user_prompt, str):

        raise TypeError(
            "user_prompt must be a string."
        )

    user_prompt = user_prompt.strip()

    if not user_prompt:

        raise ValueError(
            "user_prompt cannot be empty."
        )

    # ------------------------------------------------------
    # Ensure client exists
    # ------------------------------------------------------

    _ensure_initialized()

    # ------------------------------------------------------
    # Call Groq
    # ------------------------------------------------------

    response = client.chat.completions.create(

        model=GROQ_MODEL_NAME,

        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    # ------------------------------------------------------
    # Extract response
    # ------------------------------------------------------

    content = response.choices[0].message.content

    if content is None:

        raise RuntimeError(
            "Groq returned an empty response."
        )

    return content.strip()