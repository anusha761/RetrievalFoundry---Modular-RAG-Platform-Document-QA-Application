"""Pipeline service — orchestrates the full RAG chat pipeline.

Follows the original project's patterns:
- Each function logs entry/exit with timing
- Returns ResponseSchema where appropriate
- Tracks token costs at every LLM call
- Supports regeneration as a first-class feature
- Uses chunk_id tagging with numeric mapping for citation tracking
"""
import time
import structlog
from typing import List

from app.config import load_settings
from app.interfaces.llm_interface import LLMInterface
from app.interfaces.vector_interface import VectorStoreInterface
from app.schemas.models import PipelineState, ResponseRecord
from app.schemas.responses import ChatResponseItem, Citation, TokenCost
from app.schemas.requests import ChatRequest
from app.services import retrieval_service, summary_service, citation_service
from app.repositories import conversation_repo, response_repo
from app.utils.helpers import generate_id, get_current_time

logger = structlog.get_logger()


async def run_full_pipeline(
    request: ChatRequest,
    llm: LLMInterface,
    vector_store: VectorStoreInterface
) -> ChatResponseItem:
    """Execute the complete RAG pipeline from question to answer.

    Pipeline Steps:
        1. Record prompt (create conversation if needed, get question_id)
        2. Retrieve chunks from vector store (per file)
        3. Tag chunks with numeric chunk_ids
        4. Summarize (single-pass or batch based on file count threshold)
        5. Map chunk_ids back and generate citations
        6. Persist response to storage and update conversation pointer

    Args:
        request: The incoming chat request.
        llm: LLM service instance.
        vector_store: Vector store service instance.

    Returns:
        ChatResponseItem with summary, citations, and metadata.
    """
    settings = load_settings()
    state = PipelineState(
        question=request.question,
        file_ids=request.file_ids,
        created_date=get_current_time(),
        created_by=request.created_by.model_dump()
    )

    logger.info(f"pipeline.start - question: {state.question}, file_count: {len(state.file_ids)}")

    # --- Step 1: Record Prompt ---
    st = time.time()
    state = await step_record_prompt(state, request.conversation_id)
    et = time.time()
    logger.info(f"Time taken for step_record_prompt: {et - st:.2f} seconds")

    
    state.retriever_query = state.question

    # --- Step 3: Retrieve ---
    st = time.time()
    state = await step_retrieve(state, vector_store)
    et = time.time()
    logger.info(f"Time taken for step_retrieve: {et - st:.2f} seconds")

    # --- Step 4 & 5: Summarize (includes chunk_id tagging + citations) ---
    st = time.time()
    state = await step_summarize(state, llm, settings)
    et = time.time()
    logger.info(f"Time taken for step_summarize: {et - st:.2f} seconds")

    # --- Step 6: Persist ---
    st = time.time()
    state = await step_persist(state)
    et = time.time()
    logger.info(f"Time taken for step_persist: {et - st:.2f} seconds")

    logger.info(f"pipeline.complete - total_cost: {state.total_cost}")

    return ChatResponseItem(
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


async def step_record_prompt(state: PipelineState, conversation_id: str = None) -> PipelineState:
    """Step 1: Create conversation (if needed) and record the prompt."""
    try:
        if conversation_id:
            conv = await conversation_repo.get_conversation(conversation_id)
            if not conv:
                conv = await conversation_repo.create_conversation(state.file_ids)
        else:
            conv = await conversation_repo.create_conversation(state.file_ids)

        state.conversation_id = conv.conversation_id
        state.question_id = generate_id()

        # Get previous question for context (conversation history)
        settings = load_settings()
        history = await response_repo.get_conversation_history(
            state.conversation_id, limit=settings.CONVERSATION_HISTORY_LENGTH
        )
        if history:
            state.previous_question = history[0].prompt_question

        logger.info(f"step_record_prompt - question_id: {state.question_id}, conversation_id: {state.conversation_id}")
        return state

    except Exception as err:
        logger.error(f"Exception in step_record_prompt: {str(err)}")
        raise





async def step_retrieve(state: PipelineState, vector_store: VectorStoreInterface) -> PipelineState:
    """Step 3: Retrieve relevant chunks from vector store per file."""
    try:
        retrieved = await retrieval_service.retrieve_chunks(
            vector_store=vector_store,
            retriever_query=state.retriever_query,
            file_ids=state.file_ids
        )

        state.retrieved_chunks = retrieved
        logger.info(f"step_retrieve - files retrieved: {len(retrieved)}, total chunks: {sum(len(f['chunks']) for f in retrieved)}")
        return state

    except Exception as err:
        logger.error(f"Exception in step_retrieve: {str(err)}")
        raise


async def step_summarize(state: PipelineState, llm: LLMInterface, settings=None) -> PipelineState:
    """Step 4: Generate summary using adaptive mode (single-pass or batch).

    Includes:
    - Chunk_id numeric mapping (Approach B)
    - Tagging chunks before LLM call
    - Mapping chunk_ids back after LLM response
    - Citation generation from chunk_id references
    """
    try:
        if settings is None:
            settings = load_settings()

        domain_knowledge = settings.DOMAIN_KNOWLEDGE
        file_count = len(state.file_ids)

        # Get conversation history for context — with file overlap check
        # During regeneration, use the stored conversation_history_ids from the original response
        history_dicts = []

        if state.regenerate and state.conversation_history_ids:
            # REGENERATE: use the exact same history that was stored when the response was first created
            logger.info(f"step_summarize - regeneration mode, using stored conversation_history_ids: {state.conversation_history_ids}")
            history_records = []
            for hist_id in state.conversation_history_ids:
                record = await response_repo.get_response(hist_id)
                if record:
                    history_records.append(record)
            if history_records:
                history_dicts = [{"prompt_question": h.prompt_question, "summary": h.summary, "file_ids": h.file_ids} for h in history_records]
        else:
            # NORMAL: compute history from most recent responses in this conversation
            history = await response_repo.get_conversation_history(
                state.conversation_id, limit=settings.CONVERSATION_HISTORY_LENGTH
            )

            if history:
                # NOTE: overlap check disabled — history is now ALWAYS carried.
                # Cross-file contamination is mitigated because each history turn
                # names the files it was asked over (rendered as "Files: ...").
                # previous_file_ids = set(history[0].file_ids)
                # current_file_ids = set(state.file_ids)
                # if current_file_ids & previous_file_ids:  # intersection exists → keep history
                history_dicts = [{"prompt_question": h.prompt_question, "summary": h.summary, "file_ids": h.file_ids} for h in history]
                # Store the response_ids used as history (for future regeneration)
                state.conversation_history_ids = [h.response_id for h in history]
                logger.info(f"step_summarize - carrying history. conversation_history_ids: {state.conversation_history_ids}")
                # else:
                #     state.conversation_history_ids = []
                #     logger.info(f"step_summarize - all files changed (no overlap), skipping conversation history")
            else:
                state.conversation_history_ids = []

        if file_count <= settings.BATCH_THRESHOLD:
            # --- Single-pass mode  ---
            logger.info(f"step_summarize - SINGLE PASS mode, file_count: {file_count}")

            # Numeric-tag chunks for the single-pass LLM 
            real_to_num, num_to_real = summary_service.build_chunk_id_mapping(state.retrieved_chunks)
            tagged_chunks = summary_service.tag_chunks_with_ids(state.retrieved_chunks, real_to_num)

            st = time.time()
            parsed_summary, token_usage = await summary_service.generate_single_pass_summary(
                llm=llm,
                question=state.question,
                retrieved_chunks=tagged_chunks,
                domain_knowledge=domain_knowledge,
                conversation_history=history_dicts
            )
            et = time.time()
            logger.info(f"Time taken for single_pass_summary: {et - st:.2f} seconds")

            # Map numeric chunk_ids back to REAL ones in the answer
            answer = parsed_summary.get("ANSWER", "")
            answer = summary_service.map_chunk_ids_back(answer, num_to_real)

            state.summary = answer
            state.total_input_tokens += token_usage.get("prompt_tokens", 0)
            state.total_output_tokens += token_usage.get("completion_tokens", 0)

            # Citations: extract the REAL chunk_ids the answer referenced.
            citations = citation_service.build_citations_single_pass(
                answer_with_real_ids=answer,
                retrieved_chunks=state.retrieved_chunks,
            )
            state.citations = [c.model_dump() for c in citations]

            # Strip markers from the user-facing summary (after citation extraction)
            state.summary = summary_service.strip_citation_markers(state.summary)

        else:
            # --- Batch mode ---
            logger.info(f"step_summarize - BATCH mode, file_count: {file_count}, batch_size: {settings.BATCH_SIZE}")

            # Split the GLOBAL retrieved chunks (REAL chunk_ids, UNtagged) into batches.
            # Each batch does its OWN local numbering inside generate_batch_summary.
            batches = _split_into_batches(state.retrieved_chunks, settings.BATCH_SIZE)

            # runs batches concurrently (loop.run_in_executor + gather).
            async def _run_batch(idx, batch):
                logger.info(f"Processing batch {idx + 1}/{len(batches)}, files in batch: {len(batch)}")
                bs = time.time()
                parsed_list, batch_chunks, token_usage = await summary_service.generate_batch_summary(
                    llm=llm,
                    question=state.question,
                    batch_files=batch,
                    domain_knowledge=domain_knowledge,
                    context="",
                )
                be = time.time()
                logger.info(f"Time taken for batch {idx + 1} summary: {be - bs:.2f} seconds")
                return parsed_list, batch_chunks, token_usage

            import asyncio
            batch_results = await asyncio.gather(
                *[_run_batch(i, batch) for i, batch in enumerate(batches)]
            )

            # Flatten batch summary dicts AND aggregate per-batch processed chunks
            # in BATCH ORDER.
            all_batch_summaries = []
            aggregated_batch_chunks = []
            for parsed_list, batch_chunks, token_usage in batch_results:
                all_batch_summaries.extend(parsed_list)
                aggregated_batch_chunks.extend(batch_chunks)
                state.total_batch_input_tokens += token_usage.get("prompt_tokens", 0)
                state.total_batch_output_tokens += token_usage.get("completion_tokens", 0)

            state.batch_summaries = all_batch_summaries

            processed_file_names = [f["file_name"] for f in state.retrieved_chunks]

            # Final summary via global-renumbering chain.
            st = time.time()
            raw_final, summary_clean, numbered_pool, final_token_usage = \
                await summary_service.generate_final_summary(
                    llm=llm,
                    question=state.question,
                    batch_summaries=all_batch_summaries,
                    aggregated_batch_chunks=aggregated_batch_chunks,
                    processed_file_names=processed_file_names,
                    conversation_history=history_dicts,
                )
            et = time.time()
            logger.info(f"Time taken for final_summary: {et - st:.2f} seconds")

            state.total_input_tokens += state.total_batch_input_tokens + final_token_usage.get("prompt_tokens", 0)
            state.total_output_tokens += state.total_batch_output_tokens + final_token_usage.get("completion_tokens", 0)

            # Citations: fdi extract_citation_chunkid_pairs([n:X][c_id:Y]) from the
            # FINAL raw response, resolved via create_sourcesCollection against the
            # GLOBAL numbered pool.
            citations = citation_service.build_citations_fdi(
                final_llm_response=raw_final,
                numbered_pool=numbered_pool,
                retrieved_chunks=state.retrieved_chunks,
            )
            state.citations = [c.model_dump() for c in citations]

            # The cleaned summary (summary_json_generation + clean_markdown_table)
            # already has [n]/[c_id] markers converted to 'citation:N'. Strip any
            # residual markers for the user-facing text.
            state.summary = summary_service.strip_citation_markers(summary_clean)

        # Calculate total cost
        state.total_cost = round(
            (state.total_input_tokens * settings.INFERENCE_INPUT_COST) +
            (state.total_output_tokens * settings.INFERENCE_OUTPUT_COST),
            6
        )

        logger.info(f"step_summarize complete - total_cost: {state.total_cost}, citations: {len(state.citations)}")
        return state

    except Exception as err:
        logger.error(f"Exception in step_summarize: {str(err)}")
        raise


async def step_persist(state: PipelineState) -> PipelineState:
    """Step 6: Save response to storage and update conversation pointer."""
    try:
        record = ResponseRecord(
            response_id=generate_id(),
            conversation_id=state.conversation_id,
            question_id=state.question_id,
            prompt_question=state.question,
            summary=state.summary,
            citations=state.citations,
            file_ids=state.file_ids,
            token_cost={
                "input_tokens": state.total_input_tokens,
                "output_tokens": state.total_output_tokens,
                "total_cost": state.total_cost
            },
            conversation_history_ids=state.conversation_history_ids,
            is_regeneration=state.regenerate,
            created_by=state.created_by,
            created_date=state.created_date
        )

        await response_repo.save_response(record)
        logger.info(f"Response record inserted - response_id: {record.response_id}")

        await conversation_repo.update_conversation_last_question(
            state.conversation_id, state.question, state.question_id
        )
        logger.info(f"Conversation updated - conversation_id: {state.conversation_id}")

        return state

    except Exception as err:
        logger.error(f"Exception in step_persist: {str(err)}")
        raise


async def regenerate_response(
    conversation_id: str,
    question_id: str,
    llm: LLMInterface,
    vector_store: VectorStoreInterface,
    edited_question: str = None
) -> ChatResponseItem:
    """Re-run the pipeline for a given question (regeneration feature).

    ChatGPT-style regeneration:
    1. Find the original response by question_id
    2. Archive the old response (for audit trail)
    3. Re-run the pipeline (retrieve → summarize)
    4. Update the existing record in-place (same IDs, new content)
    5. Follow-up questions will naturally use the regenerated answer as history

    If edited_question is provided, uses the rephrased question instead of the original.
    """
    try:
        logger.info(f"regenerate_response - conversation_id: {conversation_id}, question_id: {question_id}, edited: {edited_question is not None}")

        # --- Find original response ---
        original = await response_repo.get_response_by_question_id(question_id)
        if not original:
            raise ValueError(f"No response found for question_id: {question_id}")

        logger.info(f"regenerate_response - found original response_id: {original.response_id}")

        # --- Archive the old response before overwriting (JSON file) ---
        archive_version = response_repo.get_archive_count(original.response_id) + 1
        response_repo.archive_response(original.response_id, version=archive_version, original_data=original.model_dump())
        logger.info(f"regenerate_response - archived version {archive_version} of response_id: {original.response_id}")

        # --- Use edited question if provided, otherwise original ---
        question = edited_question if edited_question else original.prompt_question

        # --- Re-run the pipeline with same file_ids ---
        settings = load_settings()
        state = PipelineState(
            question=question,
            file_ids=original.file_ids,
            conversation_id=conversation_id,
            question_id=question_id,  # Keep same question_id
            regenerate=True,
            conversation_history_ids=original.conversation_history_ids,  # Use stored history chain
            created_date=get_current_time()
        )

        
        state.retriever_query = state.question

        # Re-run retrieval
        st = time.time()
        state = await step_retrieve(state, vector_store)
        et = time.time()
        logger.info(f"Time taken for regenerate - retrieve: {et - st:.2f} seconds")

        # Re-run summarize (includes citation generation)
        st = time.time()
        state = await step_summarize(state, llm, settings)
        et = time.time()
        logger.info(f"Time taken for regenerate - summarize: {et - st:.2f} seconds")

        # --- Update existing record in-place (not insert new) ---
        update_data = {
            "summary": state.summary,
            "citations": state.citations,
            "prompt_question": question,  # Updated if edited
            "token_cost": {
                "input_tokens": state.total_input_tokens,
                "output_tokens": state.total_output_tokens,
                "total_cost": state.total_cost
            },
            "is_regeneration": True,
            "updated_date": get_current_time(),
            "regeneration_count": archive_version
        }

        modified = await response_repo.update_response(original.response_id, update_data)
        logger.info(f"regenerate_response - updated response in-place, modified: {modified}")

        # Update conversation last question pointer
        await conversation_repo.update_conversation_last_question(
            conversation_id, state.question, question_id
        )

        logger.info(f"regenerate_response complete - response_id: {original.response_id}, version: {archive_version + 1}")

        return ChatResponseItem(
            question_id=question_id,
            conversation_id=conversation_id,
            summary=state.summary,
            citations=[Citation(**c) for c in state.citations],
            token_cost=TokenCost(
                input_tokens=state.total_input_tokens,
                output_tokens=state.total_output_tokens,
                total_cost=state.total_cost
            ),
            file_ids=state.file_ids
        )

    except ValueError:
        raise
    except Exception as err:
        logger.error(f"Exception in regenerate_response: {str(err)}")
        raise


def _split_into_batches(retrieved_chunks: list, batch_size: int) -> List[list]:
    """Split retrieved chunks (grouped by file) into batches of batch_size."""
    return [
        retrieved_chunks[i:i + batch_size]
        for i in range(0, len(retrieved_chunks), batch_size)
    ]
