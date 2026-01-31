"""Benchmark runner for STaRK-Prime QA datasets using Langfuse Experiments SDK."""

# pyright: reportMissingImports=false
# pylint: disable=broad-exception-caught

import argparse
import re
import sys
from typing import Any

from langfuse import get_client, Evaluation

from stark_prime_t2s.agent.agent import (
    create_stark_prime_agent,
    run_agent_query_sync,
)
from stark_prime_t2s.config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LLM_PROVIDER,
    OPENAI_MODEL,
    OPENROUTER_MODEL,
)


def extract_node_ids_from_answer(answer: str) -> list[int]:
    """Extract node IDs from an agent's answer.

    Args:
        answer: The agent's answer text

    Returns:
        List of extracted node IDs
    """
    ids = []

    # Look for explicit ID patterns
    patterns = [
        r"(?:node\s+)?(?:IDs?|ids?)[\s:]+\[?([0-9,\s]+)\]?",  # "IDs: 1, 2, 3"
        r"(?:answer|result)[\s:]+\[?([0-9,\s]+)\]?",  # "Answer: 1, 2, 3"
        r"\b([0-9]+(?:\s*,\s*[0-9]+)+)\b",  # Comma-separated numbers
    ]

    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        for match in matches:
            try:
                extracted = [int(x.strip()) for x in match.split(",") if x.strip().isdigit()]
                ids.extend(extracted)
            except ValueError:
                continue

    # Also look for standalone numbers in the last part of the answer
    last_paragraph = answer.split("\n\n")[-1] if "\n\n" in answer else answer
    standalone_numbers = re.findall(r"\b(\d+)\b", last_paragraph)
    for num_str in standalone_numbers:
        try:
            num = int(num_str)
            # Only include reasonable node IDs (not years, counts, etc.)
            if 0 < num < 1000000 and num not in ids:
                ids.append(num)
        except ValueError:
            continue

    return ids


def compute_metrics(predicted_ids: list[int], gold_ids: list[int]) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        predicted_ids: Predicted node IDs
        gold_ids: Gold standard node IDs

    Returns:
        Dict with precision, recall, f1, and exact_match
    """
    pred_set = set(predicted_ids)
    gold_set = set(gold_ids)

    if not gold_set:
        return {
            "precision": 1.0 if not pred_set else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_set else 0.0,
            "exact_match": 1.0 if not pred_set else 0.0,
        }

    if not pred_set:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
        }

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    exact_match = 1.0 if pred_set == gold_set else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def _get_item_field(item: Any, key: str, default: Any = None) -> Any:
    """Safely get a field from a Langfuse dataset item or dict."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _normalize_expected_output(expected_output: Any) -> list[int]:
    """Normalize expected_output to a list of ints."""
    if isinstance(expected_output, list):
        return [int(x) for x in expected_output if str(x).isdigit()]
    if isinstance(expected_output, str):
        cleaned = expected_output.strip().strip("[]")
        if not cleaned:
            return []
        return [int(x.strip()) for x in cleaned.split(",") if x.strip().isdigit()]
    return []


def stark_agent_task(*, item: Any, agent: Any, **kwargs) -> str:
    """Task function that runs the STaRK-Prime agent on a question.

    Args:
        item: Dataset item with 'input' (question) and 'expected_output' (gold IDs)
        agent: The STaRK-Prime agent instance

    Returns:
        The agent's answer as a string
    """
    question = _get_item_field(item, "input", "")

    try:
        result = run_agent_query_sync(
            agent,
            question,
            trace_name=f"benchmark_{_get_item_field(item, 'id', 'unknown')}",
            tags=["benchmark", "stark-prime"],
        )
        return result.get("answer", "")
    except Exception as e:
        return f"ERROR: {e}"


def precision_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Evaluator for precision metric."""
    predicted_ids = extract_node_ids_from_answer(output)
    gold_ids = _normalize_expected_output(expected_output)
    metrics = compute_metrics(predicted_ids, gold_ids)
    return Evaluation(
        name="precision",
        value=metrics["precision"],
        comment=f"Predicted: {predicted_ids}, Gold: {gold_ids}",
    )


def recall_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Evaluator for recall metric."""
    predicted_ids = extract_node_ids_from_answer(output)
    gold_ids = _normalize_expected_output(expected_output)
    metrics = compute_metrics(predicted_ids, gold_ids)
    return Evaluation(
        name="recall",
        value=metrics["recall"],
        comment=f"Predicted: {predicted_ids}, Gold: {gold_ids}",
    )


def f1_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Evaluator for F1 metric."""
    predicted_ids = extract_node_ids_from_answer(output)
    gold_ids = _normalize_expected_output(expected_output)
    metrics = compute_metrics(predicted_ids, gold_ids)
    return Evaluation(
        name="f1",
        value=metrics["f1"],
        comment=f"Predicted: {predicted_ids}, Gold: {gold_ids}",
    )


def exact_match_evaluator(*, input, output, expected_output, **kwargs) -> Evaluation:
    """Evaluator for exact match metric."""
    predicted_ids = extract_node_ids_from_answer(output)
    gold_ids = _normalize_expected_output(expected_output)
    metrics = compute_metrics(predicted_ids, gold_ids)
    return Evaluation(
        name="exact_match",
        value=metrics["exact_match"],
        comment=f"Predicted: {predicted_ids}, Gold: {gold_ids}",
    )


def avg_precision_evaluator(*, item_results, **kwargs) -> Evaluation:
    """Run-level evaluator for average precision."""
    precisions = [
        eval.value
        for result in item_results
        for eval in result.evaluations
        if eval.name == "precision"
    ]
    if not precisions:
        return Evaluation(name="avg_precision", value=0.0)
    avg = sum(precisions) / len(precisions)
    return Evaluation(name="avg_precision", value=avg, comment=f"Average precision: {avg:.3f}")


def avg_recall_evaluator(*, item_results, **kwargs) -> Evaluation:
    """Run-level evaluator for average recall."""
    recalls = [
        eval.value
        for result in item_results
        for eval in result.evaluations
        if eval.name == "recall"
    ]
    if not recalls:
        return Evaluation(name="avg_recall", value=0.0)
    avg = sum(recalls) / len(recalls)
    return Evaluation(name="avg_recall", value=avg, comment=f"Average recall: {avg:.3f}")


def avg_f1_evaluator(*, item_results, **kwargs) -> Evaluation:
    """Run-level evaluator for average F1."""
    f1s = [
        eval.value for result in item_results for eval in result.evaluations if eval.name == "f1"
    ]
    if not f1s:
        return Evaluation(name="avg_f1", value=0.0)
    avg = sum(f1s) / len(f1s)
    return Evaluation(name="avg_f1", value=avg, comment=f"Average F1: {avg:.3f}")


def avg_exact_match_evaluator(*, item_results, **kwargs) -> Evaluation:
    """Run-level evaluator for average exact match."""
    ems = [
        eval.value
        for result in item_results
        for eval in result.evaluations
        if eval.name == "exact_match"
    ]
    if not ems:
        return Evaluation(name="avg_exact_match", value=0.0)
    avg = sum(ems) / len(ems)
    return Evaluation(
        name="avg_exact_match",
        value=avg,
        comment=f"Average exact match: {avg:.3f}",
    )


def list_available_datasets(langfuse) -> list[str]:
    """List available datasets from Langfuse.

    Args:
        langfuse: Langfuse client instance

    Returns:
        List of dataset names
    """
    try:
        response = langfuse.api.datasets.list()
        return [d.name for d in response.data]
    except Exception as e:
        print(f"Warning: Could not list datasets: {e}")
        return []


def prompt_for_dataset(langfuse) -> str | None:
    """Prompt the user to select a dataset from available datasets.

    Args:
        langfuse: Langfuse client instance

    Returns:
        Selected dataset name or None if failed
    """
    datasets = list_available_datasets(langfuse)

    if not datasets:
        print("\nNo datasets found in Langfuse. Please create datasets first.")
        print(
            "Datasets should have items with 'input' (string) and 'expected_output' (list of IDs)."
        )
        return None

    print("\n" + "=" * 60)
    print("Available Datasets:")
    print("=" * 60)
    for i, name in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    print("=" * 60)

    while True:
        choice = input("\nSelect dataset (number or name): ").strip()

        # Try to interpret as a number
        try:
            idx = int(choice)
            if 1 <= idx <= len(datasets):
                return datasets[idx - 1]
            else:
                print(f"Please enter a number between 1 and {len(datasets)}")
        except ValueError:
            # Treat as name
            if choice in datasets:
                return choice
            else:
                print(f"Dataset '{choice}' not found. Please select from the list above.")


def run_benchmark(
    dataset_name: str | None = None,
    model: str | None = None,
    concurrency: int = 1,
    verbose: bool = True,
) -> Any:
    """Run benchmark using Langfuse Experiments SDK.

    Args:
        dataset_name: Name of the Langfuse dataset to use. If None, prompts user.
        model: Model name to use (defaults to config)
        concurrency: Number of parallel evaluations
        verbose: Whether to print progress

    Returns:
        Experiment result object
    """
    # Check Langfuse credentials
    if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
        print("Error: LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY must be set")
        sys.exit(1)

    # Initialize Langfuse client
    from langfuse import Langfuse

    Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
    )
    langfuse = get_client()

    # Prompt for dataset if not provided
    if dataset_name is None:
        dataset_name = prompt_for_dataset(langfuse)
        if dataset_name is None:
            print("No dataset selected. Exiting.")
            sys.exit(1)

    # Determine the actual model name based on provider
    actual_model = model
    if not actual_model:
        actual_model = OPENROUTER_MODEL if LLM_PROVIDER == "openrouter" else OPENAI_MODEL

    if verbose:
        print(f"\nRunning benchmark on dataset: {dataset_name}")
        print(f"Provider: {LLM_PROVIDER}")
        print(f"Model: {actual_model}")
        print(f"Concurrency: {concurrency}")
        print("=" * 60)

    # Create agent
    if verbose:
        print("\nInitializing STaRK-Prime agent...")
    agent = create_stark_prime_agent(model=model)

    # Define the task with the agent bound
    def task(item, **kwargs):
        return stark_agent_task(item=item, agent=agent, **kwargs)

    # Get dataset
    try:
        dataset = langfuse.get_dataset(dataset_name)
        if verbose:
            print(f"Loaded dataset with {len(dataset.items)} items")
    except Exception as e:
        print(f"Error: Could not load dataset '{dataset_name}': {e}")
        sys.exit(1)

    # Run experiment
    if verbose:
        print("\nRunning experiment...")

    result = dataset.run_experiment(
        name=f"STaRK-Prime Benchmark - {dataset_name}",
        description=f"Benchmark run on {dataset_name} using {LLM_PROVIDER} model {actual_model}",
        task=task,
        evaluators=[
            precision_evaluator,
            recall_evaluator,
            f1_evaluator,
            exact_match_evaluator,
        ],
        run_evaluators=[
            avg_precision_evaluator,
            avg_recall_evaluator,
            avg_f1_evaluator,
            avg_exact_match_evaluator,
        ],
        max_concurrency=concurrency,
        metadata={
            "provider": LLM_PROVIDER,
            "model": actual_model,
            "dataset": dataset_name,
            "concurrency": concurrency,
        },
    )

    if verbose:
        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        print(result.format())
        print("=" * 60)

    return result


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run STaRK-Prime QA benchmark using Langfuse Experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Name of the Langfuse dataset to use (if not provided, will prompt)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (defaults to config)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of questions to run in parallel (default: 1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    # Just list datasets and exit
    if args.list_datasets:
        from langfuse import Langfuse

        if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
            print("Error: LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY must be set")
            sys.exit(1)

        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
        )
        langfuse = get_client()
        datasets = list_available_datasets(langfuse)
        print("\nAvailable datasets:")
        for name in datasets:
            print(f"  - {name}")
        sys.exit(0)

    try:
        run_benchmark(
            dataset_name=args.dataset,
            model=args.model,
            concurrency=max(1, args.concurrency),
            verbose=not args.quiet,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
