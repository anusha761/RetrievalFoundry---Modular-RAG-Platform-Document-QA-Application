"""
DeepEval RAG Evaluation

python deepeval_rag_evaluation.py --input deepeval_input.json --output deepeval_results.xlsx


INPUT JSON FORMAT:

[
    {
        "user_query": "What policies are used to manage the technology environment?",
        "retrieved_chunks": [
            "Chunk 1 content...",
            "Chunk 2 content...",
            "Chunk 3 content..."
        ],
        "final_answer": "The organization manages..."
    }
]

OUTPUT EXCEL:

Question
Final Answer
Answer Relevancy
Contextual Relevancy
Faithfulness
Answer Correctness (Needs Ground Truth. To be added later)
"""

import json
import argparse
import os
from pathlib import Path
import time

import pandas as pd

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel


# ============================================================
# CONFIGURATION
# ============================================================

# Gemini model used ONLY as the DeepEval evaluator/judge.
EVALUATOR_MODEL = "gemini-3.5-flash-lite"
# ============================================================
# RATE LIMIT HANDLING FOR FREE TIER (MAX RETRIES & THROTTLING)
# ============================================================
 # Fixes the 1.9s hard cancel timeout error
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "60"
os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "10"  # Auto-retry up to 10 times
os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = "30"   # Maximum wait time per 429 bounce

# ============================================================
# LOAD JSON
# ============================================================

def load_evaluation_data(json_path: str):

    path = Path(json_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input JSON file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Input JSON must contain a list of evaluation records."
        )

    return data


# ============================================================
# NORMALIZE RETRIEVED CHUNKS
# ============================================================

def normalize_retrieved_chunks(chunks):

    """
    Supports either:

        "retrieved_chunks": [
            "chunk text 1",
            "chunk text 2"
        ]

    OR:

        "retrieved_chunks": [
            {"content": "chunk text 1"},
            {"content": "chunk text 2"}
        ]
    """

    if not isinstance(chunks, list):
        raise ValueError(
            "retrieved_chunks must be a list."
        )

    normalized = []

    for chunk in chunks:

        if isinstance(chunk, str):
            normalized.append(chunk)

        elif isinstance(chunk, dict):

            if "content" not in chunk:
                raise ValueError(
                    "A retrieved chunk object is missing "
                    "the 'content' field."
                )

            normalized.append(str(chunk["content"]))

        else:
            raise ValueError(
                "Each retrieved chunk must be either a string "
                "or an object containing a 'content' field."
            )

    return normalized


# ============================================================
# CREATE GEMINI EVALUATOR
# ============================================================

def create_evaluator():

    """
    Gemini is used only as the LLM-as-a-judge.

    The RAG application itself continues to use Groq.
    """

    return GeminiModel(
        model=EVALUATOR_MODEL,
        temperature=0,
    )


# ============================================================
# CREATE METRICS
# ============================================================

def create_metrics(evaluator):

    answer_relevancy = AnswerRelevancyMetric(
        model=evaluator,
        include_reason=False,
    )

    contextual_relevancy = ContextualRelevancyMetric(
        model=evaluator,
        include_reason=False,
    )

    faithfulness = FaithfulnessMetric(
        model=evaluator,
        include_reason=False,
    )

    return (
        answer_relevancy,
        contextual_relevancy,
        faithfulness,
    )


# ============================================================
# RUN EVALUATION
# ============================================================

def evaluate_records(records):

    evaluator = create_evaluator()

    (
        answer_relevancy_metric,
        contextual_relevancy_metric,
        faithfulness_metric,
    ) = create_metrics(evaluator)

    results = []

    total = len(records)

    for index, record in enumerate(records, start=1):

        print(
            f"\nEvaluating question {index}/{total}..."
        )

        # ----------------------------------------------------
        # Validate input
        # ----------------------------------------------------

        required_fields = [
            "user_query",
            "retrieved_chunks",
            "final_answer",
        ]

        for field in required_fields:

            if field not in record:
                raise ValueError(
                    f"Question {index} is missing "
                    f"required field: '{field}'"
                )

        user_query = str(
            record["user_query"]
        )

        final_answer = str(
            record["final_answer"]
        )

        retrieved_chunks = normalize_retrieved_chunks(
            record["retrieved_chunks"]
        )

        # ----------------------------------------------------
        # Create DeepEval test case
        # ----------------------------------------------------

        test_case = LLMTestCase(
            input=user_query,
            actual_output=final_answer,
            retrieval_context=retrieved_chunks,
        )

        # ----------------------------------------------------
        # Answer Relevancy
        # ----------------------------------------------------

        answer_relevancy_metric.measure(
            test_case
        )

        answer_relevancy_score = (
            answer_relevancy_metric.score
        )

        print("  Pacing metric execution (6s)...")
        time.sleep(6) # Prevent spiking the RPM sensor between metrics

        # ----------------------------------------------------
        # Contextual Relevancy
        # ----------------------------------------------------

        contextual_relevancy_metric.measure(
            test_case
        )

        contextual_relevancy_score = (
            contextual_relevancy_metric.score
        )

        print("  Pacing metric execution (6s)...")
        time.sleep(6)

        # ----------------------------------------------------
        # Faithfulness
        # ----------------------------------------------------

        faithfulness_metric.measure(
            test_case
        )

        faithfulness_score = (
            faithfulness_metric.score
        )

        

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "Question": user_query,
                "Final Answer": final_answer,
                "Answer Relevancy": answer_relevancy_score,
                "Contextual Relevancy": contextual_relevancy_score,
                "Faithfulness": faithfulness_score,
            }
        )

        print(
            f"  Answer Relevancy     : "
            f"{answer_relevancy_score:.4f}"
        )

        print(
            f"  Contextual Relevancy : "
            f"{contextual_relevancy_score:.4f}"
        )

        print(
            f"  Faithfulness         : "
            f"{faithfulness_score:.4f}"
        )

        # ----------------------------------------------------
        # Extended Reset Window between rows
        # ----------------------------------------------------
        if index < total:
            print("Allowing the minute-rate window to reset. Sleeping for 20 seconds...")
            time.sleep(20)

    return results


# ============================================================
# SAVE EXCEL
# ============================================================

def save_results(results, output_path: str):

    df = pd.DataFrame(results)

    score_columns = [
        "Answer Relevancy",
        "Contextual Relevancy",
        "Faithfulness",
    ]

    for column in score_columns:
        df[column] = df[column].round(4)

    df.to_excel(
        output_path,
        index=False,
        engine="openpyxl",
    )

    print(
        f"\nEvaluation completed successfully."
    )

    print(
        f"Results saved to: {output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a RAG dataset using "
            "DeepEval and Gemini."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file.",
    )

    parser.add_argument(
        "--output",
        default="deepeval_results.xlsx",
        help="Path to output Excel file.",
    )

    args = parser.parse_args()

    records = load_evaluation_data(
        args.input
    )

    print(
        f"Loaded {len(records)} evaluation questions."
    )

    results = evaluate_records(
        records
    )

    save_results(
        results,
        args.output,
    )


if __name__ == "__main__":
    main()