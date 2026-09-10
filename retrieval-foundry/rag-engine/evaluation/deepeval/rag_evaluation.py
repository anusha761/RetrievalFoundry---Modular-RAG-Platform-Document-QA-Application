```python
"""
DeepEval RAG Evaluation

Examples:

Evaluate a specific item number:
python deepeval_rag_evaluation.py --input deepeval_input.json --item 3 --output deepeval_results.xlsx

Evaluate a specific user query:
python deepeval_rag_evaluation.py --input deepeval_input.json --user-query "What policies are used to manage the technology environment?" --output deepeval_results.xlsx


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
# RATE LIMIT HANDLING FOR FREE TIER
# ============================================================

# Fixes the 1.9s hard cancel timeout error
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "60"

# Auto-retry up to 10 times
os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "10"

# Maximum wait time per 429 bounce
os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = "30"


# ============================================================
# LOAD JSON
# ============================================================

def load_evaluation_data(json_path: str):
    """
    Load and validate the evaluation JSON file.

    Expected format:
        [
            {
                "user_query": "...",
                "retrieved_chunks": [...],
                "final_answer": "..."
            },
            ...
        ]
    """

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
# SELECT RECORD
# ============================================================

def select_record(records, item=None, user_query=None):
    """
    Select exactly one evaluation record.

    Selection can be performed using either:

        --item N
            Selects the Nth record using 1-based numbering.

        --user-query "exact query"
            Selects the record whose user_query exactly matches
            the supplied query.

    Exactly one of item or user_query must be supplied.
    """

    if item is not None and user_query is not None:
        raise ValueError(
            "Use either --item or --user-query, not both."
        )

    if item is None and user_query is None:
        raise ValueError(
            "You must specify either --item or --user-query."
        )

    # --------------------------------------------------------
    # Selection by item number
    # --------------------------------------------------------

    if item is not None:

        if item < 1:
            raise ValueError(
                "--item must be a positive integer starting from 1."
            )

        if item > len(records):
            raise IndexError(
                f"Item {item} does not exist. "
                f"The JSON file contains {len(records)} records."
            )

        selected_record = records[item - 1]

        print(
            f"Selected item {item}/{len(records)}."
        )

        return selected_record

    # --------------------------------------------------------
    # Selection by user query
    # --------------------------------------------------------

    for index, record in enumerate(records, start=1):

        if not isinstance(record, dict):
            continue

        if str(record.get("user_query", "")) == user_query:
            print(
                f"Selected item {index}/{len(records)} "
                f"using user_query."
            )

            return record

    raise ValueError(
        "No record found with the specified user_query."
    )


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

def evaluate_record(record, item_number=None):

    evaluator = create_evaluator()

    (
        answer_relevancy_metric,
        contextual_relevancy_metric,
        faithfulness_metric,
    ) = create_metrics(evaluator)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    required_fields = [
        "user_query",
        "retrieved_chunks",
        "final_answer",
    ]

    for field in required_fields:

        if field not in record:
            item_label = (
                f"Item {item_number}"
                if item_number is not None
                else "Selected record"
            )

            raise ValueError(
                f"{item_label} is missing "
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

    # --------------------------------------------------------
    # Display selected question
    # --------------------------------------------------------

    if item_number is not None:
        print(
            f"\nEvaluating item {item_number}..."
        )
    else:
        print("\nEvaluating selected question...")

    print(f"Question: {user_query}")

    # --------------------------------------------------------
    # Create DeepEval test case
    # --------------------------------------------------------

    test_case = LLMTestCase(
        input=user_query,
        actual_output=final_answer,
        retrieval_context=retrieved_chunks,
    )

    # --------------------------------------------------------
    # Answer Relevancy
    # --------------------------------------------------------

    print("\nRunning Answer Relevancy...")

    answer_relevancy_metric.measure(
        test_case
    )

    answer_relevancy_score = (
        answer_relevancy_metric.score
    )

    print(
        f"  Answer Relevancy : "
        f"{answer_relevancy_score:.4f}"
    )

    print("  Pacing metric execution (6s)...")
    time.sleep(6)

    # --------------------------------------------------------
    # Contextual Relevancy
    # --------------------------------------------------------

    print("\nRunning Contextual Relevancy...")

    contextual_relevancy_metric.measure(
        test_case
    )

    contextual_relevancy_score = (
        contextual_relevancy_metric.score
    )

    print(
        f"  Contextual Relevancy : "
        f"{contextual_relevancy_score:.4f}"
    )

    print("  Pacing metric execution (6s)...")
    time.sleep(6)

    # --------------------------------------------------------
    # Faithfulness
    # --------------------------------------------------------

    print("\nRunning Faithfulness...")

    faithfulness_metric.measure(
        test_case
    )

    faithfulness_score = (
        faithfulness_metric.score
    )

    print(
        f"  Faithfulness : "
        f"{faithfulness_score:.4f}"
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {
        "Question": user_query,
        "Final Answer": final_answer,
        "Answer Relevancy": answer_relevancy_score,
        "Contextual Relevancy": contextual_relevancy_score,
        "Faithfulness": faithfulness_score,
    }

    return [result]


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
            "Evaluate one selected RAG record using "
            "DeepEval and Gemini."
        )
    )

    # --------------------------------------------------------
    # Input JSON
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input JSON file.",
    )

    # --------------------------------------------------------
    # Output Excel
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        default="deepeval_results.xlsx",
        help="Path to output Excel file.",
    )

    # --------------------------------------------------------
    # Selection by item number
    # --------------------------------------------------------

    parser.add_argument(
        "--item",
        type=int,
        help=(
            "1-based item number to evaluate. "
            "Example: --item 3"
        ),
    )

    # --------------------------------------------------------
    # Selection by user query
    # --------------------------------------------------------

    parser.add_argument(
        "--user-query",
        help=(
            "Exact user_query value of the record to evaluate."
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load JSON
    # --------------------------------------------------------

    records = load_evaluation_data(
        args.input
    )

    print(
        f"Loaded {len(records)} evaluation questions."
    )

    # --------------------------------------------------------
    # Select exactly one record
    # --------------------------------------------------------

    selected_record = select_record(
        records=records,
        item=args.item,
        user_query=args.user_query,
    )

    # Determine item number for display
    selected_item_number = None

    if args.item is not None:
        selected_item_number = args.item

    else:
        for index, record in enumerate(records, start=1):
            if record is selected_record:
                selected_item_number = index
                break

    # --------------------------------------------------------
    # Evaluate selected record only
    # --------------------------------------------------------

    results = evaluate_record(
        record=selected_record,
        item_number=selected_item_number,
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    save_results(
        results,
        args.output,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
```
