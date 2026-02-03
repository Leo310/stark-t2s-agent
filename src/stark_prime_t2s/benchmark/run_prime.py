"""Benchmark runner for STaRK-Prime QA datasets using a custom Langfuse loop."""

# pyright: reportMissingImports=false
# pylint: disable=broad-exception-caught

import argparse
import concurrent.futures
import random
import sys
import threading
from typing import Any

from langfuse import get_client
from tqdm import tqdm
from tqdm import tqdm as _tqdm_module

from stark_prime_t2s.agent.agent import (
    create_stark_prime_agent,
    create_stark_prime_entity_resolver_agent,
    create_stark_prime_sparql_agent,
    create_stark_prime_sql_agent,
    get_langfuse_handler,
    run_agent_query_sync,
)
from stark_prime_t2s.config import (
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LLM_PROVIDER,
    OPENAI_MODEL,
    OPENROUTER_MODEL,
)


def compute_metrics(predicted_ids: list[int], gold_ids: list[int]) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        predicted_ids: Predicted node IDs (order matters for ranking metrics)
        gold_ids: Gold standard node IDs

    Returns:
        Dict with precision, recall, f1, exact_match, hit@1, hit@5, and mrr
    """
    pred_set = set(predicted_ids)
    gold_set = set(gold_ids)

    if not gold_set:
        return {
            "precision": 1.0 if not pred_set else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not pred_set else 0.0,
            "exact_match": 1.0 if not pred_set else 0.0,
            "hit@1": 1.0 if not pred_set else 0.0,
            "hit@5": 1.0 if not pred_set else 0.0,
            "mrr": 1.0 if not pred_set else 0.0,
        }

    if not pred_set:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "hit@1": 0.0,
            "hit@5": 0.0,
            "mrr": 0.0,
        }

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    exact_match = 1.0 if pred_set == gold_set else 0.0

    # Hit@K metrics: check if any gold ID appears in top K predictions
    hit_at_1 = 0.0
    hit_at_5 = 0.0
    mrr = 0.0

    # Check hit@1: is the first prediction correct?
    if predicted_ids and predicted_ids[0] in gold_set:
        hit_at_1 = 1.0

    # Check hit@5: is any gold ID in the top 5 predictions?
    top_5_preds = predicted_ids[:5]
    if any(pred_id in gold_set for pred_id in top_5_preds):
        hit_at_5 = 1.0

    # Compute MRR: reciprocal of the rank of the first correct prediction
    for rank, pred_id in enumerate(predicted_ids, start=1):
        if pred_id in gold_set:
            mrr = 1.0 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
        "hit@1": hit_at_1,
        "hit@5": hit_at_5,
        "mrr": mrr,
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


def stark_agent_task(*, item: Any, agent: Any, **kwargs) -> dict[str, Any]:
    """Task function that runs the STaRK-Prime agent on a question.

    Args:
        item: Dataset item with 'input' (question) and 'expected_output' (gold IDs)
        agent: The STaRK-Prime agent instance

    Returns:
        Dict with 'node_ids' and 'reasoning' keys
    """
    question = _get_item_field(item, "input", "")

    result = run_agent_query_sync(
        agent,
        question,
        trace_name=f"benchmark_{_get_item_field(item, 'id', 'unknown')}",
        tags=["benchmark", "stark-prime"],
    )

    return {
        "node_ids": result["node_ids"],
        "reasoning": result["reasoning"],
    }


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
    agent_mode: str = "auto",
) -> Any:
    """Run benchmark using a custom loop with Langfuse dataset runs.

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

    agent_mode_normalized = agent_mode.strip().lower()

    def agent_factory(build_index: bool):
        if agent_mode_normalized == "sql":
            return create_stark_prime_sql_agent(
                model=model,
                build_entity_index_on_start=build_index,
            )
        if agent_mode_normalized == "sparql":
            return create_stark_prime_sparql_agent(
                model=model,
                build_entity_index_on_start=build_index,
            )
        if agent_mode_normalized == "entity":
            return create_stark_prime_entity_resolver_agent(
                model=model,
                build_entity_index_on_start=build_index,
            )
        return create_stark_prime_agent(
            model=model,
            build_entity_index_on_start=build_index,
        )

    agent = agent_factory(build_index=True)

    # Initialize Langfuse tracing before progress bar output
    _ = get_langfuse_handler()

    # Get dataset
    try:
        dataset = langfuse.get_dataset(dataset_name)
        if verbose:
            print(f"Loaded dataset with {len(dataset.items)} items")
    except Exception as e:
        print(f"Error: Could not load dataset '{dataset_name}': {e}")
        sys.exit(1)

    if concurrency > 1 and verbose:
        print(f"Running with thread concurrency: {concurrency}")

    # Prompt for run name
    default_run_name = f"STaRK-Prime Benchmark - {dataset_name}"
    run_name = input(f"\nRun name [{default_run_name}]: ").strip() or default_run_name
    run_description = f"Benchmark run on {dataset_name} using {LLM_PROVIDER} model {actual_model}"
    run_metadata = {
        "provider": LLM_PROVIDER,
        "model": actual_model,
        "dataset": dataset_name,
        "concurrency": concurrency,
    }

    if verbose:
        print("\nRunning benchmark loop...")

    item_metrics: list[dict[str, float]] = []
    failures: list[dict[str, str]] = []

    items_all = list(dataset.items)
    random.shuffle(items_all)

    required_scores = {
        "precision",
        "recall",
        "f1",
        "exact_match",
        "hit@1",
        "hit@5",
        "mrr",
    }

    def _fetch_existing_run_ids() -> tuple[set[str], set[str]]:
        try:
            run_with_items = langfuse.get_dataset_run(
                dataset_name=dataset_name,
                run_name=run_name,
            )
        except Exception as exc:
            if "NotFound" in str(exc):
                return set(), set()
            if verbose:
                print(f"Warning: Could not fetch dataset run: {exc}")
            return set(), set()

        completed: set[str] = set()
        failed: set[str] = set()

        for run_item in run_with_items.dataset_run_items:
            item_id = str(run_item.dataset_item_id)
            if item_id in completed:
                continue
            try:
                trace = langfuse.api.trace.get(run_item.trace_id)
                score_names = {getattr(score, "name", None) for score in trace.scores}
                if required_scores.issubset(score_names):
                    completed.add(item_id)
                else:
                    failed.add(item_id)
            except Exception:
                failed.add(item_id)

        return completed, failed

    completed_ids, failed_ids = _fetch_existing_run_ids()
    if completed_ids or failed_ids:
        if verbose:
            missing = len(items_all) - len(completed_ids)
            print(
                "Resuming from Langfuse dataset run: "
                f"completed {len(completed_ids)}, failed {len(failed_ids)}, missing {missing}"
            )

    items = [
        item
        for item in items_all
        if str(_get_item_field(item, "id", "unknown")) not in completed_ids
    ]
    if not items and not failed_ids:
        if verbose:
            print("All dataset items already completed for this run name.")
        return {
            "run_name": run_name,
            "metrics": {},
            "items": 0,
            "failures": [],
        }
    progress = tqdm(
        items,
        disable=not verbose,
        desc="Benchmarking",
        unit="item",
        dynamic_ncols=True,
        mininterval=0.5,
    )

    thread_local = threading.local()
    stop_event = threading.Event()

    def _get_thread_agent():
        agent_instance = getattr(thread_local, "agent", None)
        if agent_instance is None:
            agent_instance = agent_factory(build_index=False)
            thread_local.agent = agent_instance
        return agent_instance

    def _process_item(item):
        if stop_event.is_set():
            raise RuntimeError("Benchmark interrupted")
        item_id = _get_item_field(item, "id", "unknown")
        question = _get_item_field(item, "input", "")
        expected_output = _get_item_field(item, "expected_output", None)
        gold_ids = _normalize_expected_output(expected_output)

        with item.run(
            run_name=run_name,
            run_description=run_description,
            run_metadata=run_metadata,
        ) as root_span:
            output = stark_agent_task(item=item, agent=_get_thread_agent())

            root_span.update_trace(input=question, output=output)

            metrics = compute_metrics(output["node_ids"], gold_ids)
            for name, value in metrics.items():
                root_span.score_trace(
                    name=name,
                    value=value,
                    comment=f"Predicted: {output['node_ids']}, Gold: {gold_ids}",
                )

        return item_id, metrics

    future_to_item: dict[concurrent.futures.Future, Any] | None = None

    try:
        if concurrency <= 1:
            for item in progress:
                try:
                    _, metrics = _process_item(item)
                    item_metrics.append(metrics)
                except Exception as exc:
                    item_id = _get_item_field(item, "id", "unknown")
                    failures.append({"item_id": str(item_id), "error": str(exc)})
                    if verbose:
                        _tqdm_module.write(f"Warning: item {item_id} failed: {exc}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_item = {executor.submit(_process_item, item): item for item in items}
                for future in concurrent.futures.as_completed(future_to_item):
                    progress.update(1)
                    try:
                        _, metrics = future.result()
                        item_metrics.append(metrics)
                    except Exception as exc:
                        item = future_to_item[future]
                        item_id = _get_item_field(item, "id", "unknown")
                        failures.append({"item_id": str(item_id), "error": str(exc)})
                        if verbose:
                            _tqdm_module.write(f"Warning: item {item_id} failed: {exc}")
    except KeyboardInterrupt:
        stop_event.set()
        if verbose:
            _tqdm_module.write("Benchmark interrupted. Cancelling pending tasks...")
        if concurrency > 1 and future_to_item:
            for future in list(future_to_item.keys()):
                if not future.done():
                    future.cancel()
        raise

    langfuse.flush()

    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    avg_metrics = {
        "avg_precision": _avg([m["precision"] for m in item_metrics]),
        "avg_recall": _avg([m["recall"] for m in item_metrics]),
        "avg_f1": _avg([m["f1"] for m in item_metrics]),
        "avg_exact_match": _avg([m["exact_match"] for m in item_metrics]),
        "avg_hit@1": _avg([m["hit@1"] for m in item_metrics]),
        "avg_hit@5": _avg([m["hit@5"] for m in item_metrics]),
        "avg_mrr": _avg([m["mrr"] for m in item_metrics]),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Benchmark Complete!")
        print("=" * 60)
        print(f"Items processed: {len(item_metrics)}")
        print(f"Failures: {len(failures)}")
        for name, value in avg_metrics.items():
            print(f"{name}: {value:.3f}")
        if failures:
            print("\nFailed items:")
            for failure in failures[:10]:
                print(f"  - {failure['item_id']}: {failure['error']}")
            if len(failures) > 10:
                print(f"  ... {len(failures) - 10} more")
        print("=" * 60)

    return {
        "run_name": run_name,
        "metrics": avg_metrics,
        "items": len(item_metrics),
        "failures": failures,
    }


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
        "--agent",
        type=str,
        default=None,
        choices=["auto", "sql", "sparql", "entity"],
        help="Agent mode: auto (SQL+SPARQL), sql-only, sparql-only, or entity-only",
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
        if args.agent:
            agent_mode = args.agent
        else:
            prompt = "Select agent mode [auto/sql/sparql/entity] (default: auto): "
            choice = input(prompt).strip().lower()
            agent_mode = choice or "auto"
            if agent_mode not in ("auto", "sql", "sparql", "entity"):
                print("Invalid selection. Using auto mode.")
                agent_mode = "auto"
        run_benchmark(
            dataset_name=args.dataset,
            model=args.model,
            concurrency=max(1, args.concurrency),
            verbose=not args.quiet,
            agent_mode=agent_mode,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
