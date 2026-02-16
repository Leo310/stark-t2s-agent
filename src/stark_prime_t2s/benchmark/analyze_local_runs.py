"""Analyze local benchmark run artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from string import Template
from typing import Any

_TEMPLATE_DIR = os.path.dirname(os.path.abspath(__file__))


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = line.strip()
            if not record:
                continue
            items.append(json.loads(record))
    return items


# Regex patterns for extracting tool parameters from content
_SEARCH_ENTITIES_RESULT_PATTERN = re.compile(r"Found (\d+) entities")

# Patterns for detecting zero/empty results
_SQL_SPARQL_NO_RESULTS_PATTERNS = [
    re.compile(r"no\s+results?\s+found", re.IGNORECASE),
    re.compile(r"empty\s+result", re.IGNORECASE),
    re.compile(r"0\s+rows?\s+returned", re.IGNORECASE),
    re.compile(r"query\s+returned\s+no", re.IGNORECASE),
]
_SEARCH_ZERO_RESULTS_PATTERN = re.compile(r"Found 0 entities")

# Regex patterns for extracting entity IDs from tool outputs
_SEARCH_ENTITY_ID_PATTERN = re.compile(r"\(ID:\s*(\d+)\)")
# Match standalone numeric IDs on their own line
_SQL_SPARQL_STANDALONE_ID_PATTERN = re.compile(r"^\s*(\d+)\s*$", re.MULTILINE)
# Match IDs at the start of a table row (e.g., "128507 | name | ...")
_SQL_SPARQL_TABLE_ROW_ID_PATTERN = re.compile(r"^\s*(\d+)\s*\|", re.MULTILINE)
# Match IDs in pipe-separated values (any column that's just a number)
_SQL_SPARQL_PIPE_ID_PATTERN = re.compile(r"\|\s*(\d+)\s*(?:\||$)", re.MULTILINE)
# Match IDs in URI format (e.g., "http://stark.stanford.edu/prime/node/4900")
_URI_NODE_ID_PATTERN = re.compile(r"/node/(\d+)")

# Regex patterns for detecting tool errors in output
# Matches "Error executing sql query: <message>" or "Error executing sparql query: <message>"
_TOOL_ERROR_PATTERN = re.compile(
    r"Error executing (?:sql|sparql) query:\s*(.+?)(?:\n|$)", re.IGNORECASE
)
# Matches "SPARQL query error: <message>" from Fuseki
_SPARQL_ERROR_PATTERN = re.compile(r"SPARQL query error:\s*(.+?)(?:\n|$)", re.IGNORECASE)
# Common error categories for classification
# Note: Order matters - more specific patterns should come first
_ERROR_CATEGORIES = {
    # "forbidden keyword" errors are actually syntax issues (reserved words used incorrectly)
    "syntax": re.compile(
        r"syntax|parse|QueryBadFormed|malformed|forbidden keyword|unexpected|invalid", re.IGNORECASE
    ),
    "not_found": re.compile(
        r"does not exist|not found|unknown|undefined|no such|unrecognized|missing", re.IGNORECASE
    ),
    "permission": re.compile(
        r"permission|denied|access denied|unauthorized|forbidden", re.IGNORECASE
    ),
    "timeout": re.compile(r"timeout|timed out|too long|exceeded|limit", re.IGNORECASE),
    "connection": re.compile(
        r"connection|connect|network|unreachable|refused|reset", re.IGNORECASE
    ),
    "type_error": re.compile(
        r"type\s*error|cannot\s*cast|incompatible|conversion|datatype", re.IGNORECASE
    ),
    "constraint": re.compile(
        r"constraint|violation|duplicate|unique|foreign\s*key|integrity", re.IGNORECASE
    ),
    "resource": re.compile(
        r"memory|out\s*of|overflow|too\s*many|resource|exhausted", re.IGNORECASE
    ),
}


def _extract_tool_error(tool_name: str, content: str) -> dict[str, Any] | None:
    """Extract error information from tool call output.

    Args:
        tool_name: Name of the tool
        content: Tool output content

    Returns:
        Dict with error info if an error was found, None otherwise:
        - message: The error message
        - category: Categorized error type (syntax, not_found, permission, timeout, connection, other)
        - tool: The tool name
    """
    if not content:
        return None

    error_msg = None

    # Check for explicit error patterns
    match = _TOOL_ERROR_PATTERN.search(content)
    if match:
        error_msg = match.group(1).strip()
    else:
        match = _SPARQL_ERROR_PATTERN.search(content)
        if match:
            error_msg = match.group(1).strip()

    if not error_msg:
        return None

    # Categorize the error
    category = "other"
    for cat_name, cat_pattern in _ERROR_CATEGORIES.items():
        if cat_pattern.search(error_msg):
            category = cat_name
            break

    return {
        "message": error_msg,
        "category": category,
        "tool": tool_name,
    }


def _analyze_tool_errors(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze tool errors from a list of tool calls.

    Args:
        tool_calls: List of tool call dicts with name and content

    Returns a dict with:
    - total_calls: total number of tool calls analyzed
    - error_count: number of tool calls with errors
    - error_rate: fraction of tool calls with errors
    - by_tool: {tool_name: {total: N, errors: N, rate: float}}
    - by_category: {category: count}
    - errors: list of error details [{message, category, tool}, ...]
    """
    result: dict[str, Any] = {
        "total_calls": 0,
        "error_count": 0,
        "error_rate": 0.0,
        "by_tool": {},
        "by_category": {},
        "errors": [],
    }

    tool_stats: dict[str, dict[str, int]] = {}
    category_counts: dict[str, int] = {}

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue

        tool_name = tc.get("name", "unknown")
        content = tc.get("content", "") or ""

        # Only analyze query tools
        if tool_name not in (
            "search_entities_tool",
            "execute_sql_query_tool",
            "execute_sparql_query_tool",
        ):
            continue

        result["total_calls"] += 1

        if tool_name not in tool_stats:
            tool_stats[tool_name] = {"total": 0, "errors": 0}

        tool_stats[tool_name]["total"] += 1

        error_info = _extract_tool_error(tool_name, content)
        if error_info:
            result["error_count"] += 1
            tool_stats[tool_name]["errors"] += 1
            result["errors"].append(error_info)

            category = error_info["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

    # Calculate rates
    if result["total_calls"] > 0:
        result["error_rate"] = result["error_count"] / result["total_calls"]

    for tool_name, stats in tool_stats.items():
        rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
        result["by_tool"][tool_name] = {
            "total": stats["total"],
            "errors": stats["errors"],
            "rate": rate,
        }

    result["by_category"] = category_counts

    return result


def _is_zero_result_query(tool_name: str, content: str) -> bool:
    """Check if a tool call returned zero results.

    Different tools have different ways of indicating empty results:
    - search_entities_tool: "Found 0 entities"
    - execute_sql_query_tool / execute_sparql_query_tool: various patterns
    """
    if not content:
        return True  # Empty content is considered zero results

    if tool_name == "search_entities_tool":
        return bool(_SEARCH_ZERO_RESULTS_PATTERN.search(content))
    elif tool_name in ("execute_sql_query_tool", "execute_sparql_query_tool"):
        # Check explicit "no results" patterns
        for pattern in _SQL_SPARQL_NO_RESULTS_PATTERNS:
            if pattern.search(content):
                return True
        # Also check if no IDs were extracted (empty result set)
        ids = _extract_entity_ids_from_content(tool_name, content)
        # Consider it zero results if content is short and has no IDs
        # (handles cases where query succeeded but returned empty table)
        if len(ids) == 0 and len(content.strip()) < 200:
            return True
    return False


def _extract_entity_ids_from_content(tool_name: str, content: str) -> set[int]:
    """Extract entity IDs mentioned in tool call output.

    Different tools have different output formats:
    - search_entities_tool: "... (ID: 12345) ..."
    - execute_sql_query_tool / execute_sparql_query_tool: rows with numeric IDs
      - Standalone: just a number on its own line
      - Table format: "12345 | name | ..." or "... | 12345 | ..."
    """
    ids: set[int] = set()

    if not content:
        return ids

    if tool_name == "search_entities_tool":
        # Search results format: "(ID: 12345)"
        for match in _SEARCH_ENTITY_ID_PATTERN.finditer(content):
            try:
                ids.add(int(match.group(1)))
            except ValueError:
                continue
    elif tool_name in ("execute_sql_query_tool", "execute_sparql_query_tool"):
        # SQL/SPARQL results can have multiple formats:
        # 1. Standalone numeric IDs on their own line
        for match in _SQL_SPARQL_STANDALONE_ID_PATTERN.finditer(content):
            try:
                ids.add(int(match.group(1)))
            except ValueError:
                continue
        # 2. IDs at start of table rows: "128507 | name | ..."
        for match in _SQL_SPARQL_TABLE_ROW_ID_PATTERN.finditer(content):
            try:
                ids.add(int(match.group(1)))
            except ValueError:
                continue
        # 3. IDs in any pipe-separated column: "| 128507 |" or "| 128507"
        for match in _SQL_SPARQL_PIPE_ID_PATTERN.finditer(content):
            try:
                ids.add(int(match.group(1)))
            except ValueError:
                continue
        # 4. IDs in URI format: "http://.../node/12345"
        for match in _URI_NODE_ID_PATTERN.finditer(content):
            try:
                ids.add(int(match.group(1)))
            except ValueError:
                continue

    return ids


def _analyze_zero_result_queries(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze how often tool queries returned zero results.

    Args:
        tool_calls: List of tool call dicts with name and content

    Returns a dict with:
    - total_queries: total number of tool calls analyzed
    - zero_result_queries: number of queries that returned 0 results
    - zero_result_rate: fraction of queries with 0 results
    - by_tool: {tool_name: {total: N, zero: N, rate: float}}
    """
    result: dict[str, Any] = {
        "total_queries": 0,
        "zero_result_queries": 0,
        "zero_result_rate": 0.0,
        "by_tool": {},
    }

    tool_stats: dict[str, dict[str, int]] = {}

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue

        tool_name = tc.get("name", "unknown")
        content = tc.get("content", "") or ""

        # Only analyze query tools
        if tool_name not in (
            "search_entities_tool",
            "execute_sql_query_tool",
            "execute_sparql_query_tool",
        ):
            continue

        result["total_queries"] += 1

        if tool_name not in tool_stats:
            tool_stats[tool_name] = {"total": 0, "zero": 0}

        tool_stats[tool_name]["total"] += 1

        if _is_zero_result_query(tool_name, content):
            result["zero_result_queries"] += 1
            tool_stats[tool_name]["zero"] += 1

    # Calculate rates
    if result["total_queries"] > 0:
        result["zero_result_rate"] = result["zero_result_queries"] / result["total_queries"]

    for tool_name, stats in tool_stats.items():
        rate = stats["zero"] / stats["total"] if stats["total"] > 0 else 0.0
        result["by_tool"][tool_name] = {
            "total": stats["total"],
            "zero": stats["zero"],
            "rate": rate,
        }

    return result


def _analyze_ground_truth_discovery(
    tool_calls: list[dict[str, Any]],
    gold_ids: set[int],
    predicted_ids: set[int] | None = None,
) -> dict[str, Any]:
    """Analyze which tools discovered ground truth entities and when.

    Args:
        tool_calls: List of tool call dicts with name and content
        gold_ids: Set of ground truth entity IDs
        predicted_ids: Set of entity IDs returned by the agent (optional)

    Returns a dict with:
    - first_discovery: {tool_name, call_index} for first tool that found any gold ID
    - by_tool: {tool_name: {found_ids: [...], call_indices: [...]}}
    - discovery_order: [(tool_name, call_index, found_id), ...] in order of discovery
    - coverage: fraction of gold_ids found by any tool
    - found_but_not_returned: gold IDs seen in tool output but not in agent's response
    - returned_correct: gold IDs both found by tools and returned by agent
    - missed_opportunity_rate: fraction of found gold IDs that weren't returned
    """
    result: dict[str, Any] = {
        "first_discovery": None,
        "by_tool": {},
        "discovery_order": [],
        "found_ids": [],
        "not_found_ids": list(gold_ids),
        "coverage": 0.0,
        # New fields for comparing with predicted output
        "found_but_not_returned": [],
        "returned_correct": [],
        "missed_opportunity_rate": 0.0,
        # New fields for tool output analysis
        "total_ids_from_tools": 0,
        "unique_ids_from_tools": [],
        "tool_precision": 0.0,  # gold_found / total_unique_ids
        # Cumulative recall at each tool call index
        "recall_by_call_index": [],  # [(call_index, recall, {tool: recall_contribution}), ...]
        # Cumulative precision at each tool call index
        "precision_by_call_index": [],  # [(call_index, precision, {tool: precision_contribution}), ...]
    }

    if not gold_ids:
        return result

    all_found: set[int] = set()
    all_tool_ids: set[int] = set()  # Track ALL IDs returned by tools
    seen_discoveries: set[int] = set()  # Track already-discovered IDs
    # Track which tool found each gold ID (first tool to find it gets credit)
    gold_found_by_tool: dict[str, set[int]] = {}  # {tool_name: set of gold IDs found by this tool}
    # Track all IDs returned by each tool (for precision calculation)
    all_ids_by_tool: dict[str, set[int]] = {}  # {tool_name: set of all IDs returned by this tool}
    recall_progression: list[
        tuple[int, float, dict[str, float]]
    ] = []  # Track recall after each iteration
    precision_progression: list[
        tuple[int, float, dict[str, float]]
    ] = []  # Track precision after each iteration

    # Group tool calls by iteration
    # If iteration info is not available, fall back to treating each call as its own iteration
    calls_by_iteration: dict[int, list[dict[str, Any]]] = {}
    for idx, tc in enumerate(tool_calls):
        if not isinstance(tc, dict):
            continue
        # Use iteration from tool call if available, otherwise use sequential index
        iteration = tc.get("iteration", idx + 1)
        if iteration not in calls_by_iteration:
            calls_by_iteration[iteration] = []
        calls_by_iteration[iteration].append(tc)

    # Process iterations in order
    for iteration in sorted(calls_by_iteration.keys()):
        iteration_calls = calls_by_iteration[iteration]

        for tc in iteration_calls:
            tool_name = tc.get("name", "unknown")
            content = tc.get("content", "") or ""

            # Extract IDs from this tool call
            found_in_call = _extract_entity_ids_from_content(tool_name, content)
            all_tool_ids.update(found_in_call)  # Track all IDs from tools

            # Track all IDs returned by each tool (for precision)
            if tool_name not in all_ids_by_tool:
                all_ids_by_tool[tool_name] = set()
            all_ids_by_tool[tool_name].update(found_in_call)

            gold_found = found_in_call & gold_ids

            if gold_found:
                # Track by tool
                if tool_name not in result["by_tool"]:
                    result["by_tool"][tool_name] = {"found_ids": [], "call_indices": []}

                new_finds = gold_found - seen_discoveries
                if new_finds:
                    result["by_tool"][tool_name]["found_ids"].extend(sorted(new_finds))
                    result["by_tool"][tool_name]["call_indices"].append(iteration)

                    # Track which tool gets credit for finding each gold ID
                    if tool_name not in gold_found_by_tool:
                        gold_found_by_tool[tool_name] = set()
                    gold_found_by_tool[tool_name].update(new_finds)

                    # Track discovery order (only new discoveries)
                    for gid in sorted(new_finds):
                        result["discovery_order"].append((tool_name, iteration, gid))
                        seen_discoveries.add(gid)

                    # Track first discovery
                    if result["first_discovery"] is None:
                        result["first_discovery"] = {
                            "tool_name": tool_name,
                            "call_index": iteration,
                        }

                all_found.update(gold_found)

        # Track cumulative recall and precision after this iteration completes
        # Only record if this iteration had query tools
        has_query_tools = any(
            tc.get("name")
            in (
                "search_entities_tool",
                "execute_sql_query_tool",
                "execute_sparql_query_tool",
            )
            for tc in iteration_calls
        )
        if has_query_tools:
            # Recall = gold_found / total_gold
            current_recall = len(all_found) / len(gold_ids)
            # Calculate recall contribution by each tool
            recall_contributions: dict[str, float] = {}
            for tn, found_set in gold_found_by_tool.items():
                recall_contributions[tn] = len(found_set) / len(gold_ids)
            recall_progression.append((iteration, current_recall, recall_contributions))

            # Precision = gold_found / total_ids_returned
            current_precision = len(all_found) / len(all_tool_ids) if all_tool_ids else 0.0
            # Calculate precision contribution by each tool (gold found by tool / all IDs by tool)
            precision_contributions: dict[str, float] = {}
            for tn, found_set in gold_found_by_tool.items():
                tool_total_ids = len(all_ids_by_tool.get(tn, set()))
                if tool_total_ids > 0:
                    precision_contributions[tn] = len(found_set) / tool_total_ids
                else:
                    precision_contributions[tn] = 0.0
            precision_progression.append((iteration, current_precision, precision_contributions))

    result["recall_by_call_index"] = recall_progression
    result["precision_by_call_index"] = precision_progression

    result["found_ids"] = sorted(all_found)
    result["not_found_ids"] = sorted(gold_ids - all_found)
    result["coverage"] = len(all_found) / len(gold_ids) if gold_ids else 0.0

    # Tool output statistics
    result["total_ids_from_tools"] = len(all_tool_ids)
    result["unique_ids_from_tools"] = sorted(all_tool_ids)
    result["tool_precision"] = len(all_found) / len(all_tool_ids) if all_tool_ids else 0.0

    # Compare with predicted IDs if provided
    if predicted_ids is not None:
        predicted_correct = predicted_ids & gold_ids
        found_and_correct = all_found & gold_ids

        # Gold IDs that tools found but agent didn't return
        found_but_not_returned = found_and_correct - predicted_correct
        result["found_but_not_returned"] = sorted(found_but_not_returned)

        # Gold IDs that were both found by tools and returned by agent
        returned_correct = found_and_correct & predicted_correct
        result["returned_correct"] = sorted(returned_correct)

        # Missed opportunity rate: of the gold IDs tools found, how many did agent miss?
        if found_and_correct:
            result["missed_opportunity_rate"] = len(found_but_not_returned) / len(found_and_correct)

    return result


def _extract_tool_params(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Extract inferred parameters from tool call content when input is not recorded.

    This function parses tool output to infer parameters like top_k from search results.
    """
    params: dict[str, Any] = {}
    name = tool_call.get("name", "")
    content = tool_call.get("content", "") or ""

    # If input is available, use it directly
    if tool_call.get("input"):
        return tool_call["input"]

    # Infer parameters from content based on tool type
    if name == "search_entities_tool":
        # Extract top_k from "Found N entities" pattern
        match = _SEARCH_ENTITIES_RESULT_PATTERN.search(content)
        if match:
            params["top_k"] = int(match.group(1))

    return params


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[int(k)]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def _aggregate_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    latencies: list[float] = []
    tool_calls = Counter()

    for item in items:
        metrics = item.get("metrics", {}) or {}
        for name, value in metrics.items():
            try:
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
                metric_counts[name] = metric_counts.get(name, 0) + 1
            except (TypeError, ValueError):
                continue

        timestamps = item.get("timestamps", {}) or {}
        latency = timestamps.get("latency_s")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        for tool_call in item.get("tool_calls", []) or []:
            name = tool_call.get("name") if isinstance(tool_call, dict) else str(tool_call)
            if name:
                tool_calls[name] += 1

    metrics_avg = {
        f"avg_{name}": (metric_sums[name] / metric_counts[name]) if metric_counts[name] else 0.0
        for name in metric_sums
    }

    latency_stats = {
        "count": len(latencies),
        "min": min(latencies) if latencies else 0.0,
        "p50": _percentile(latencies, 50),
        "p90": _percentile(latencies, 90),
        "p95": _percentile(latencies, 95),
        "max": max(latencies) if latencies else 0.0,
        "avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
    }

    return {
        "metrics": metrics_avg,
        "latency_s": latency_stats,
        "tool_call_counts": dict(tool_calls.most_common()),
    }


def _compute_histogram(values: list[float], bins: int = 20) -> dict[str, Any]:
    if not values:
        return {"bins": [], "counts": []}
    if bins <= 0:
        bins = 1
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return {"bins": [f"{min_value:.3f}"], "counts": [len(values)]}
    width = (max_value - min_value) / bins
    edges = [min_value + i * width for i in range(bins + 1)]
    counts = [0 for _ in range(bins)]
    for value in values:
        if value == max_value:
            counts[-1] += 1
            continue
        index = int((value - min_value) / width)
        index = max(0, min(index, bins - 1))
        counts[index] += 1
    labels = [f"{edges[i]:.3f}-{edges[i + 1]:.3f}" for i in range(bins)]
    return {"bins": labels, "counts": counts}


def _metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "p50": _percentile(values, 50),
        "p90": _percentile(values, 90),
        "p95": _percentile(values, 95),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def _aggregate_tool_params(items: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """Aggregate tool parameter distributions across all items.

    Returns:
        Dict mapping tool_name -> param_name -> {value: count}
        Example: {"search_entities_tool": {"top_k": {"5": 100, "10": 50, "15": 30}}}
    """
    tool_param_dist: dict[str, dict[str, dict[str, int]]] = {}

    for item in items:
        tool_calls = item.get("tool_calls", []) or []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tool_name = tc.get("name", "unknown")
            params = _extract_tool_params(tc)
            for param_name, param_value in params.items():
                # Convert value to string for counting
                value_str = str(param_value)
                tool_param_dist.setdefault(tool_name, {}).setdefault(param_name, {})
                tool_param_dist[tool_name][param_name][value_str] = (
                    tool_param_dist[tool_name][param_name].get(value_str, 0) + 1
                )

    return tool_param_dist


def _aggregate_tool_calls_by_iteration(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tool call distribution by iteration across all items.

    Returns:
        Dict with:
        - iterations: list of iteration numbers
        - by_tool: {tool_name: [count_at_iter_1, count_at_iter_2, ...]}
        - totals: [total_calls_at_iter_1, total_calls_at_iter_2, ...]
        - max_iteration: maximum iteration number observed
        - items_by_iteration: [num_items_reaching_iter_1, ...]
    """
    # Collect tool calls by iteration across all items
    iteration_tool_counts: dict[int, dict[str, int]] = {}  # {iter: {tool: count}}
    items_reaching_iteration: dict[int, int] = {}  # {iter: num_items}
    all_tools: set[str] = set()
    max_iteration = 0

    for item in items:
        tool_calls = item.get("tool_calls", []) or []
        if not tool_calls:
            continue

        # Group tool calls by iteration for this item
        item_iter_tools: dict[int, dict[str, int]] = {}
        item_max_iter = 0

        for idx, tc in enumerate(tool_calls):
            if not isinstance(tc, dict):
                continue
            tool_name = tc.get("name", "unknown")
            # Use iteration from tool call if available, otherwise fall back to sequential index
            # This matches the fallback behavior in _analyze_ground_truth_discovery
            iteration = tc.get("iteration", idx + 1)
            all_tools.add(tool_name)
            item_max_iter = max(item_max_iter, iteration)

            if iteration not in item_iter_tools:
                item_iter_tools[iteration] = {}
            item_iter_tools[iteration][tool_name] = item_iter_tools[iteration].get(tool_name, 0) + 1

        # Track max iteration
        max_iteration = max(max_iteration, item_max_iter)

        # Track which iterations this item reached
        for iter_num in range(1, item_max_iter + 1):
            items_reaching_iteration[iter_num] = items_reaching_iteration.get(iter_num, 0) + 1

        # Aggregate tool counts by iteration
        for iter_num, tool_counts in item_iter_tools.items():
            if iter_num not in iteration_tool_counts:
                iteration_tool_counts[iter_num] = {}
            for tool_name, count in tool_counts.items():
                iteration_tool_counts[iter_num][tool_name] = (
                    iteration_tool_counts[iter_num].get(tool_name, 0) + count
                )

    # Build output structure
    iterations = list(range(1, max_iteration + 1))
    sorted_tools = sorted(all_tools)

    by_tool: dict[str, list[int]] = {tool: [] for tool in sorted_tools}
    totals: list[int] = []
    items_by_iteration: list[int] = []

    for iter_num in iterations:
        iter_counts = iteration_tool_counts.get(iter_num, {})
        iter_total = 0
        for tool_name in sorted_tools:
            count = iter_counts.get(tool_name, 0)
            by_tool[tool_name].append(count)
            iter_total += count
        totals.append(iter_total)
        items_by_iteration.append(items_reaching_iteration.get(iter_num, 0))

    return {
        "iterations": iterations,
        "by_tool": by_tool,
        "totals": totals,
        "max_iteration": max_iteration,
        "items_by_iteration": items_by_iteration,
    }


def _aggregate_tool_errors_from_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tool error statistics across all items.

    Args:
        items: List of benchmark item dicts (each containing tool_calls)

    Returns:
        Dict with aggregated error statistics:
        - total_calls: total number of query tool calls across all items
        - error_count: total number of errors across all items
        - error_rate: overall error rate
        - by_tool: {tool_name: {total: N, errors: N, rate: float}}
        - by_category: {category: count}
        - items_with_errors: number of items that had at least one tool error
        - items_with_errors_rate: fraction of items with at least one error
    """
    total_calls = 0
    error_count = 0
    tool_stats: dict[str, dict[str, int]] = {}
    category_counts: dict[str, int] = {}
    items_with_errors = 0

    for item in items:
        tool_calls = item.get("tool_calls", []) or []
        item_error_stats = _analyze_tool_errors(tool_calls)

        total_calls += item_error_stats["total_calls"]
        error_count += item_error_stats["error_count"]

        if item_error_stats["error_count"] > 0:
            items_with_errors += 1

        # Aggregate by tool
        for tool_name, stats in item_error_stats["by_tool"].items():
            if tool_name not in tool_stats:
                tool_stats[tool_name] = {"total": 0, "errors": 0}
            tool_stats[tool_name]["total"] += stats["total"]
            tool_stats[tool_name]["errors"] += stats["errors"]

        # Aggregate by category
        for category, count in item_error_stats["by_category"].items():
            category_counts[category] = category_counts.get(category, 0) + count

    # Calculate rates
    error_rate = error_count / total_calls if total_calls > 0 else 0.0
    items_with_errors_rate = items_with_errors / len(items) if items else 0.0

    by_tool: dict[str, dict[str, Any]] = {}
    for tool_name, stats in tool_stats.items():
        rate = stats["errors"] / stats["total"] if stats["total"] > 0 else 0.0
        by_tool[tool_name] = {
            "total": stats["total"],
            "errors": stats["errors"],
            "rate": rate,
        }

    return {
        "total_calls": total_calls,
        "error_count": error_count,
        "error_rate": error_rate,
        "by_tool": by_tool,
        "by_category": category_counts,
        "items_with_errors": items_with_errors,
        "items_with_errors_rate": items_with_errors_rate,
    }


def _aggregate_zero_results_from_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate zero-result query statistics across all items.

    Args:
        items: List of benchmark item dicts (each containing tool_calls)

    Returns:
        Dict with aggregated zero-result statistics:
        - total_queries: total number of query tool calls across all items
        - zero_result_queries: total number of zero-result queries
        - zero_result_rate: overall zero-result rate
        - by_tool: {tool_name: {total: N, zero: N, rate: float}}
        - items_with_zero_results: number of items that had at least one zero-result query
        - items_with_zero_results_rate: fraction of items with zero-result queries
    """
    total_queries = 0
    zero_result_queries = 0
    tool_stats: dict[str, dict[str, int]] = {}
    items_with_zero_results = 0

    for item in items:
        tool_calls = item.get("tool_calls", []) or []
        item_zero_stats = _analyze_zero_result_queries(tool_calls)

        total_queries += item_zero_stats["total_queries"]
        zero_result_queries += item_zero_stats["zero_result_queries"]

        if item_zero_stats["zero_result_queries"] > 0:
            items_with_zero_results += 1

        # Aggregate by tool
        for tool_name, stats in item_zero_stats["by_tool"].items():
            if tool_name not in tool_stats:
                tool_stats[tool_name] = {"total": 0, "zero": 0}
            tool_stats[tool_name]["total"] += stats["total"]
            tool_stats[tool_name]["zero"] += stats["zero"]

    # Calculate rates
    zero_result_rate = zero_result_queries / total_queries if total_queries > 0 else 0.0
    items_with_zero_results_rate = items_with_zero_results / len(items) if items else 0.0

    by_tool: dict[str, dict[str, Any]] = {}
    for tool_name, stats in tool_stats.items():
        rate = stats["zero"] / stats["total"] if stats["total"] > 0 else 0.0
        by_tool[tool_name] = {
            "total": stats["total"],
            "zero": stats["zero"],
            "rate": rate,
        }

    return {
        "total_queries": total_queries,
        "zero_result_queries": zero_result_queries,
        "zero_result_rate": zero_result_rate,
        "by_tool": by_tool,
        "items_with_zero_results": items_with_zero_results,
        "items_with_zero_results_rate": items_with_zero_results_rate,
    }


def _build_report_data(
    items: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    aggregates: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    latencies: list[float] = []
    tool_call_counts: list[int] = []
    per_metric_values: dict[str, list[float]] = {}

    for item in items:
        metrics = item.get("metrics", {}) or {}
        for name, value in metrics.items():
            try:
                per_metric_values.setdefault(name, []).append(float(value))
            except (TypeError, ValueError):
                continue

        timestamps = item.get("timestamps", {}) or {}
        latency = timestamps.get("latency_s")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))

        tool_calls = item.get("tool_calls", []) or []
        tool_call_counts.append(len(tool_calls))

    tool_call_count_dist = Counter(tool_call_counts)
    tool_call_count_labels = [str(count) for count in sorted(tool_call_count_dist.keys())]
    tool_call_count_values = [tool_call_count_dist[int(label)] for label in tool_call_count_labels]

    # Aggregate tool parameter distributions
    tool_param_distributions = _aggregate_tool_params(items)

    # Aggregate tool calls by iteration
    tool_calls_by_iteration = _aggregate_tool_calls_by_iteration(items)

    # Aggregate tool error statistics
    tool_error_stats = _aggregate_tool_errors_from_items(items)

    # Aggregate zero-result query statistics
    zero_result_stats = _aggregate_zero_results_from_items(items)

    metric_summaries = {name: _metric_stats(values) for name, values in per_metric_values.items()}
    metric_histograms = {
        name: _compute_histogram(values, bins=20) for name, values in per_metric_values.items()
    }

    failure_errors = []
    for failure in failures:
        if isinstance(failure, dict):
            error = failure.get("error") or "Unknown error"
            failure_errors.append(str(error))
        else:
            failure_errors.append(str(failure))
    failure_error_counts = Counter(failure_errors)

    return {
        "run_metadata": summary.get("run_metadata", {}),
        "items": len(items),
        "failures": len(failures),
        "failure_details": failures,
        "failure_error_counts": dict(failure_error_counts.most_common()),
        "item_rows": _collect_item_rows(items, failures),
        "latency": {
            "stats": aggregates.get("latency_s", {}),
            "histogram": _compute_histogram(latencies, bins=20),
        },
        "tool_calls": {
            "name_counts": aggregates.get("tool_call_counts", {}),
            "count_distribution": {
                "labels": tool_call_count_labels,
                "counts": tool_call_count_values,
            },
            "param_distributions": tool_param_distributions,
            "by_iteration": tool_calls_by_iteration,
        },
        "tool_errors": tool_error_stats,
        "zero_result_queries": zero_result_stats,
        "metrics": {
            "averages": aggregates.get("metrics", {}),
            "summaries": metric_summaries,
            "histograms": metric_histograms,
        },
    }


def _write_html_report(report: dict[str, Any], output_path: str) -> None:
    report_json = json.dumps(report, ensure_ascii=True)
    template_path = os.path.join(_TEMPLATE_DIR, "report_template.html")
    with open(template_path, "r", encoding="utf-8") as handle:
        template = Template(handle.read())
    html = template.safe_substitute(report_json=report_json)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _write_json_report(report: dict[str, Any], output_path: str) -> None:
    """Write report data to a JSON file for programmatic consumption.

    The JSON file contains all the same data visible in the HTML report,
    including run metadata, metrics, per-item details, tool analysis,
    and multi-dataset comparisons when applicable.
    """
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)


def _gold_size_bucket(gold_count: int) -> str:
    """Bucket gold answer set size into categories."""
    if gold_count <= 1:
        return "1"
    if gold_count <= 5:
        return "2-5"
    if gold_count <= 20:
        return "6-20"
    return "20+"


def _query_complexity(query: str) -> str:
    """Heuristic query complexity from text patterns.

    Looks at entity mentions (capitalized multi-word sequences), logical
    connectives, and overall length to bucket into simple / moderate / complex.
    """
    if not query:
        return "simple"
    # Count likely entity mentions (sequences of Capitalized words / alphanumeric codes)
    entity_pattern = re.compile(r"\b[A-Z][A-Za-z0-9]{1,}")
    entities = entity_pattern.findall(query)
    entity_count = len(entities)
    # Count logical connectives / constraints
    connectives = len(
        re.findall(r"\b(?:and|or|but|not|both|either|neither|as well as)\b", query, re.IGNORECASE)
    )
    word_count = len(query.split())

    score = entity_count + connectives * 1.5 + (word_count / 10)
    if score >= 6:
        return "complex"
    if score >= 3:
        return "moderate"
    return "simple"


def _tool_profile(tool_counts: dict[str, int]) -> str:
    """Derive tool usage profile from per-item tool counts."""
    has_sparql = any("sparql" in k.lower() for k in tool_counts)
    has_sql = any("sql" in k.lower() and "sparql" not in k.lower() for k in tool_counts)
    has_search = any("search" in k.lower() for k in tool_counts)

    parts: list[str] = []
    if has_sql:
        parts.append("SQL")
    if has_sparql:
        parts.append("SPARQL")
    if has_search:
        parts.append("Search")
    return "+".join(parts) if parts else "none"


def _latency_bucket(latency_s: float | None, thresholds: tuple[float, float] = (30.0, 90.0)) -> str:
    """Bucket latency into fast / medium / slow."""
    if latency_s is None or math.isnan(latency_s):
        return "unknown"
    if latency_s <= thresholds[0]:
        return "fast (<=30s)"
    if latency_s <= thresholds[1]:
        return "medium (30-90s)"
    return "slow (>90s)"


def _collect_item_rows(
    items: list[dict[str, Any]], failures: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    rows = []
    for item in items:
        metrics = item.get("metrics", {}) or {}
        # Count tool usage per item and extract parameters
        tool_calls = item.get("tool_calls", []) or []
        tool_counts: dict[str, int] = {}
        tool_params: dict[str, list[dict[str, Any]]] = {}  # {tool_name: [params_dict, ...]}
        # Track tool calls by iteration for this item: {iteration: {tool_name: count}}
        tool_calls_by_iter: dict[int, dict[str, int]] = {}
        max_iteration = 0
        for idx, tc in enumerate(tool_calls):
            if isinstance(tc, dict):
                tool_name = tc.get("name", "unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                # Extract and store tool parameters
                params = _extract_tool_params(tc)
                if params:
                    tool_params.setdefault(tool_name, []).append(params)
                # Track by iteration (consistent fallback with other functions)
                iteration = tc.get("iteration", idx + 1)
                max_iteration = max(max_iteration, iteration)
                if iteration not in tool_calls_by_iter:
                    tool_calls_by_iter[iteration] = {}
                tool_calls_by_iter[iteration][tool_name] = (
                    tool_calls_by_iter[iteration].get(tool_name, 0) + 1
                )

        # Extract gold IDs and latency
        gold_ids_raw = item.get("gold_ids") or item.get("expected_output") or []
        gold_ids_list = gold_ids_raw if isinstance(gold_ids_raw, list) else []
        gold_ids_set = set(gold_ids_list)
        gold_count = len(gold_ids_list)
        timestamps = item.get("timestamps", {}) or {}
        latency_s = timestamps.get("latency_s")
        if isinstance(latency_s, (int, float)):
            latency_s = float(latency_s)
        else:
            latency_s = None

        query = item.get("input") or ""

        # Extract predicted IDs from agent output
        output = item.get("output") or {}
        predicted_ids_raw = output.get("node_ids") if isinstance(output, dict) else None
        predicted_ids_set: set[int] | None = None
        if predicted_ids_raw and isinstance(predicted_ids_raw, list):
            predicted_ids_set = set(predicted_ids_raw)

        # Analyze ground truth discovery (now with predicted IDs comparison)
        gt_discovery = _analyze_ground_truth_discovery(tool_calls, gold_ids_set, predicted_ids_set)

        # Analyze zero-result queries
        zero_result_stats = _analyze_zero_result_queries(tool_calls)

        # Analyze tool errors
        tool_error_stats = _analyze_tool_errors(tool_calls)

        # Derive categories
        categories = {
            "gold_size": _gold_size_bucket(gold_count),
            "query_complexity": _query_complexity(query),
            "tool_profile": _tool_profile(tool_counts),
            "latency_bucket": _latency_bucket(latency_s),
        }

        # Extract reasoning from agent output
        reasoning = output.get("reasoning") if isinstance(output, dict) else None

        # Build detailed tool calls array for Query Details section
        tool_calls_detail = [
            {
                "name": tc.get("name", "unknown"),
                "input": tc.get("input"),
                "content": tc.get("content", ""),
                "iteration": tc.get("iteration", idx + 1),
            }
            for idx, tc in enumerate(tool_calls)
            if isinstance(tc, dict)
        ]

        row = {
            "item_id": item.get("item_id"),
            "input": query,
            "metrics": metrics,
            "tool_counts": tool_counts,
            "tool_params": tool_params,
            "tool_calls_total": len(tool_calls),
            "tool_calls_by_iter": tool_calls_by_iter,  # {iteration: {tool_name: count}}
            "max_iteration": max_iteration,
            "gold_count": gold_count,
            "latency_s": latency_s,
            "categories": categories,
            "gt_discovery": gt_discovery,
            "zero_result_stats": zero_result_stats,
            "tool_error_stats": tool_error_stats,
            "failed": False,
            # NEW: Fields for Query Details section
            "reasoning": reasoning,
            "predicted_ids": list(predicted_ids_raw) if isinstance(predicted_ids_raw, list) else [],
            "gold_ids": gold_ids_list,
            "tool_calls_detail": tool_calls_detail,
        }
        rows.append(row)
    # Include failures with a failed flag
    for failure in failures or []:
        if isinstance(failure, dict):
            gold_ids = failure.get("gold_ids") or failure.get("expected_output") or []
            gold_count = len(gold_ids) if isinstance(gold_ids, list) else 0
            query = failure.get("input") or ""
            categories = {
                "gold_size": _gold_size_bucket(gold_count),
                "query_complexity": _query_complexity(query),
                "tool_profile": "none",
                "latency_bucket": "unknown",
            }
            gold_ids_list = gold_ids if isinstance(gold_ids, list) else []
            row = {
                "item_id": failure.get("item_id"),
                "input": query,
                "metrics": {},
                "tool_counts": {},
                "tool_params": {},
                "tool_calls_total": 0,
                "tool_calls_by_iter": {},
                "max_iteration": 0,
                "gold_count": gold_count,
                "latency_s": None,
                "categories": categories,
                "gt_discovery": {
                    "first_discovery": None,
                    "by_tool": {},
                    "discovery_order": [],
                    "found_ids": [],
                    "not_found_ids": [],
                    "coverage": 0.0,
                    "found_but_not_returned": [],
                    "returned_correct": [],
                    "missed_opportunity_rate": 0.0,
                    "total_ids_from_tools": 0,
                    "unique_ids_from_tools": [],
                    "tool_precision": 0.0,
                    "recall_by_call_index": [],
                    "precision_by_call_index": [],
                },
                "zero_result_stats": {
                    "total_queries": 0,
                    "zero_result_queries": 0,
                    "zero_result_rate": 0.0,
                    "by_tool": {},
                },
                "tool_error_stats": {
                    "total_calls": 0,
                    "error_count": 0,
                    "error_rate": 0.0,
                    "by_tool": {},
                    "by_category": {},
                    "errors": [],
                },
                "failed": True,
                "error": failure.get("error"),
                # NEW: Fields for Query Details section (empty for failures)
                "reasoning": None,
                "predicted_ids": [],
                "gold_ids": gold_ids_list,
                "tool_calls_detail": [],
            }
            rows.append(row)
    return rows


def _collect_run_dirs(compare_dir: str) -> list[str]:
    if not compare_dir or not os.path.isdir(compare_dir):
        return []
    run_dirs = []
    for entry in os.listdir(compare_dir):
        candidate = os.path.join(compare_dir, entry)
        if not os.path.isdir(candidate):
            continue
        items_path = os.path.join(candidate, "items.jsonl")
        if os.path.exists(items_path):
            run_dirs.append(candidate)
    return run_dirs


def _select_latest_run(compare_dir: str) -> str:
    run_dirs = _collect_run_dirs(compare_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No runs with items.jsonl found in {compare_dir}")
    latest_dir = max(run_dirs, key=lambda path: os.path.getmtime(os.path.join(path, "items.jsonl")))
    return latest_dir


def _summarize_run_dir(run_dir: str) -> dict[str, Any]:
    items_path = os.path.join(run_dir, "items.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    items = _read_jsonl(items_path)
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

    aggregates = _aggregate_metrics(items)
    return {
        "run_name": os.path.basename(run_dir),
        "run_path": run_dir,
        "items": len(items),
        "failures": len(_read_jsonl(os.path.join(run_dir, "failures.jsonl")))
        if os.path.exists(os.path.join(run_dir, "failures.jsonl"))
        else 0,
        "latency": aggregates.get("latency_s", {}),
        "metrics": aggregates.get("metrics", {}),
        "run_metadata": summary.get("run_metadata", {}),
    }


def _collect_run_summaries(compare_dir: str) -> list[dict[str, Any]]:
    return [_summarize_run_dir(path) for path in sorted(_collect_run_dirs(compare_dir))]


def _label_to_key(label: str) -> str:
    """Convert a display label to a simple key.

    Examples:
        "Human-Generated Dataset" -> "human"
        "Synthesized Dataset (100)" -> "synthesized"
    """
    lower = label.lower()
    if "human" in lower:
        return "human"
    if "synth" in lower:
        return "synthesized"
    # Fallback: slugify the label
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def _cleanup_stale_failures(run_dir: str) -> int:
    """Remove stale failures from failures.jsonl that now exist in items.jsonl.

    Args:
        run_dir: Path to the benchmark run directory.

    Returns:
        Number of stale failures removed.
    """
    items_path = os.path.join(run_dir, "items.jsonl")
    failures_path = os.path.join(run_dir, "failures.jsonl")

    if not os.path.exists(failures_path) or not os.path.exists(items_path):
        return 0

    # Get completed item IDs
    completed_ids: set[str] = set()
    with open(items_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                item_id = str(record.get("item_id", ""))
                if item_id:
                    completed_ids.add(item_id)
            except (json.JSONDecodeError, KeyError):
                continue

    # Read failures and filter out stale ones
    remaining_failures: list[dict[str, Any]] = []
    stale_count = 0
    with open(failures_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                item_id = str(record.get("item_id", ""))
                if item_id and item_id in completed_ids:
                    stale_count += 1
                else:
                    remaining_failures.append(record)
            except (json.JSONDecodeError, KeyError):
                continue

    if stale_count > 0:
        # Rewrite failures.jsonl
        with open(failures_path, "w", encoding="utf-8") as handle:
            for record in remaining_failures:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    return stale_count


def cleanup_all_stale_failures(compare_dir: str | None = None) -> dict[str, int]:
    """Clean up stale failures across all runs in a directory.

    Args:
        compare_dir: Directory containing benchmark runs. Defaults to ./benchmark_runs.

    Returns:
        Dict mapping run name to number of stale failures removed.
    """
    compare_dir = compare_dir or os.path.join(os.getcwd(), "benchmark_runs")
    results: dict[str, int] = {}
    for run_path in _collect_run_dirs(compare_dir):
        run_name = os.path.basename(run_path)
        removed = _cleanup_stale_failures(run_path)
        if removed > 0:
            results[run_name] = removed
    return results


def analyze_run(
    run_dir: str | None,
    output_dir: str | None = None,
    include_html: bool = True,
    include_json: bool = True,
    compare_dir: str | None = None,
    cleanup_failures: bool = True,
    additional_compare_dirs: dict[str, str] | None = None,
) -> dict[str, Any]:
    if run_dir is None:
        compare_dir = compare_dir or os.path.join(os.getcwd(), "benchmark_runs")
        run_dir = _select_latest_run(compare_dir)
    if compare_dir is None:
        compare_dir = os.path.dirname(run_dir)

    # Clean up stale failures across all runs before analysis
    if cleanup_failures:
        cleanup_results = cleanup_all_stale_failures(compare_dir)
        for name, count in cleanup_results.items():
            print(f"Cleaned up {count} stale failure(s) in {name}")

    items_path = os.path.join(run_dir, "items.jsonl")
    failures_path = os.path.join(run_dir, "failures.jsonl")
    summary_path = os.path.join(run_dir, "summary.json")

    if not os.path.exists(items_path):
        raise FileNotFoundError(f"items.jsonl not found in {run_dir}")

    output_dir = output_dir or compare_dir
    os.makedirs(output_dir, exist_ok=True)

    items = _read_jsonl(items_path)
    failures = []
    if os.path.exists(failures_path):
        failures = _read_jsonl(failures_path)

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)

    aggregates = _aggregate_metrics(items)
    aggregates["items"] = len(items)
    aggregates["failures"] = len(failures)
    aggregates["run_metadata"] = summary.get("run_metadata", {})

    results: dict[str, Any] = {
        "items": len(items),
        "failures": len(failures),
    }

    # Build report data if either HTML or JSON output is requested
    if include_html or include_json:
        report_data = _build_report_data(items, failures, aggregates, summary)
        report_data["compare_runs"] = _collect_run_summaries(compare_dir)
        report_data["primary_run"] = os.path.basename(run_dir)
        report_data["all_runs"] = {}
        for run_path in _collect_run_dirs(compare_dir):
            run_items = _read_jsonl(os.path.join(run_path, "items.jsonl"))
            run_failures = []
            run_failures_path = os.path.join(run_path, "failures.jsonl")
            if os.path.exists(run_failures_path):
                run_failures = _read_jsonl(run_failures_path)
            run_summary = {}
            run_summary_path = os.path.join(run_path, "summary.json")
            if os.path.exists(run_summary_path):
                with open(run_summary_path, "r", encoding="utf-8") as handle:
                    run_summary = json.load(handle)
            run_aggregates = _aggregate_metrics(run_items)
            run_report = _build_report_data(run_items, run_failures, run_aggregates, run_summary)
            run_report["run_name"] = os.path.basename(run_path)
            report_data["all_runs"][os.path.basename(run_path)] = run_report

        # Handle multi-dataset mode
        if additional_compare_dirs:
            datasets: dict[str, Any] = {}

            # Primary dataset (from compare_dir)
            primary_label = "Human-Generated Dataset"
            primary_key = _label_to_key(primary_label)
            datasets[primary_key] = {
                "key": primary_key,
                "label": primary_label,
                "compare_runs": report_data["compare_runs"],
                "all_runs": report_data["all_runs"],
                "primary_run": report_data["primary_run"],
            }

            # Additional datasets
            for label, path in additional_compare_dirs.items():
                if cleanup_failures:
                    cleanup_results = cleanup_all_stale_failures(path)
                    for name, count in cleanup_results.items():
                        print(f"Cleaned up {count} stale failure(s) in {name}")

                key = _label_to_key(label)
                add_compare_runs = _collect_run_summaries(path)
                add_all_runs: dict[str, Any] = {}
                add_primary_run = None

                for add_run_path in _collect_run_dirs(path):
                    add_run_name = os.path.basename(add_run_path)
                    add_run_items = _read_jsonl(os.path.join(add_run_path, "items.jsonl"))
                    add_run_failures = []
                    add_run_failures_path = os.path.join(add_run_path, "failures.jsonl")
                    if os.path.exists(add_run_failures_path):
                        add_run_failures = _read_jsonl(add_run_failures_path)
                    add_run_summary = {}
                    add_run_summary_path = os.path.join(add_run_path, "summary.json")
                    if os.path.exists(add_run_summary_path):
                        with open(add_run_summary_path, "r", encoding="utf-8") as handle:
                            add_run_summary = json.load(handle)
                    add_run_aggregates = _aggregate_metrics(add_run_items)
                    add_run_report = _build_report_data(
                        add_run_items, add_run_failures, add_run_aggregates, add_run_summary
                    )
                    add_run_report["run_name"] = add_run_name
                    add_all_runs[add_run_name] = add_run_report

                    if add_primary_run is None:
                        add_primary_run = add_run_name

                datasets[key] = {
                    "key": key,
                    "label": label,
                    "compare_runs": add_compare_runs,
                    "all_runs": add_all_runs,
                    "primary_run": add_primary_run,
                }

            # Add multi-dataset fields to report
            report_data["datasets"] = datasets
            report_data["default_dataset"] = primary_key
            report_data["has_multiple_datasets"] = True

        # Write HTML report
        if include_html:
            html_out = os.path.join(output_dir, "analysis_report.html")
            _write_html_report(report_data, html_out)
            results["html_report"] = html_out

        # Write JSON report
        if include_json:
            json_out = os.path.join(output_dir, "report.json")
            _write_json_report(report_data, json_out)
            results["json_report"] = json_out

            # Write individual JSON reports for each dataset
            dataset_json_reports: dict[str, str] = {}

            # Write JSON for primary dataset
            primary_dataset_report = {
                "dataset_key": _label_to_key("Human-Generated Dataset"),
                "dataset_label": "Human-Generated Dataset",
                "compare_runs": report_data["compare_runs"],
                "all_runs": report_data["all_runs"],
                "primary_run": report_data["primary_run"],
            }
            primary_json_out = os.path.join(compare_dir, "report.json")
            _write_json_report(primary_dataset_report, primary_json_out)
            dataset_json_reports["human"] = primary_json_out

            # Write JSON for additional datasets
            if additional_compare_dirs:
                for label, path in additional_compare_dirs.items():
                    key = _label_to_key(label)
                    if key in report_data.get("datasets", {}):
                        dataset_data = report_data["datasets"][key]
                        dataset_report = {
                            "dataset_key": key,
                            "dataset_label": label,
                            "compare_runs": dataset_data["compare_runs"],
                            "all_runs": dataset_data["all_runs"],
                            "primary_run": dataset_data["primary_run"],
                        }
                        dataset_json_out = os.path.join(path, "report.json")
                        _write_json_report(dataset_report, dataset_json_out)
                        dataset_json_reports[key] = dataset_json_out

            results["dataset_json_reports"] = dataset_json_reports

    return results


def _get_dataset_label(dir_name: str) -> str:
    """Get a human-readable label for a benchmark directory.

    Known directories get specific labels, others get a formatted version of the dir name.
    """
    # Known dataset directory mappings
    known_labels = {
        "benchmark_runs": "Human-Generated Dataset",
        "benchmark_runs_human": "Human-Generated Dataset",
        "benchmark_runs_synth": "Synthesized Dataset (100)",
    }

    if dir_name in known_labels:
        return known_labels[dir_name]

    # Format unknown directory names: benchmark_runs_foo -> "Foo Dataset"
    name = dir_name.replace("benchmark_runs_", "").replace("benchmark_runs", "").strip("_")
    if name:
        return name.replace("_", " ").title() + " Dataset"
    return dir_name


def _discover_benchmark_dirs(base_path: str, filters: list[str] | None = None) -> dict[str, str]:
    """Discover benchmark directories in the given base path.

    Args:
        base_path: Directory to search for benchmark_runs* directories
        filters: Optional list of directory name patterns to include (substring match)

    Returns:
        Dict mapping labels to directory paths, ordered with human-generated first
    """
    discovered: dict[str, str] = {}

    # Look for directories matching benchmark_runs* pattern
    for entry in os.listdir(base_path):
        entry_path = os.path.join(base_path, entry)
        if not os.path.isdir(entry_path):
            continue
        if not entry.startswith("benchmark_runs"):
            continue

        # Check if this directory has any run subdirs (with items.jsonl)
        if not _collect_run_dirs(entry_path):
            continue

        # Apply filters if provided
        if filters:
            # Check if any filter matches this directory name
            if not any(f.lower() in entry.lower() for f in filters):
                continue

        label = _get_dataset_label(entry)
        discovered[label] = entry_path

    # Sort to ensure consistent ordering: Human-Generated first, then alphabetical
    def sort_key(item: tuple[str, str]) -> tuple[int, str]:
        label, _ = item
        if "Human" in label:
            return (0, label)
        if "Synth" in label:
            return (1, label)
        return (2, label)

    return dict(sorted(discovered.items(), key=sort_key))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze local STaRK-Prime benchmark run artifacts. "
            "When no arguments are given, auto-discovers all benchmark_runs* directories "
            "and generates a combined report with a dataset selector."
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help=(
            "Dataset directory names to include (substring match). "
            "If not specified, all benchmark_runs* directories are included. "
            "Examples: 'human' matches benchmark_runs, 'synth' matches benchmark_runs_synth"
        ),
    )
    parser.add_argument(
        "--run-dir",
        required=False,
        help="Path to specific benchmark run directory (contains items.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis files (defaults to current dir for multi-dataset, or dataset dir for single)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Disable HTML report generation",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON report generation",
    )
    args = parser.parse_args()

    # Auto-discover benchmark directories
    base_path = os.getcwd()
    filters = args.datasets if args.datasets else None
    discovered = _discover_benchmark_dirs(base_path, filters)

    if not discovered:
        print("No benchmark_runs* directories found with valid runs.")
        if filters:
            print(f"Filters applied: {filters}")
        return

    print(f"Discovered {len(discovered)} dataset(s):")
    for label, path in discovered.items():
        run_count = len(_collect_run_dirs(path))
        print(f"  - {label}: {path} ({run_count} runs)")

    # If only one dataset found, use single-dataset mode
    if len(discovered) == 1:
        label, path = next(iter(discovered.items()))
        output_dir = args.output_dir or path
        results = analyze_run(
            args.run_dir,
            output_dir,
            include_html=not args.no_html,
            include_json=not args.no_json,
            compare_dir=path,
            additional_compare_dirs=None,
        )
        print(f"Analyzed {results['items']} items, {results['failures']} failures")
        if "html_report" in results:
            print(f"Wrote HTML report: {results['html_report']}")
        if "json_report" in results:
            print(f"Wrote JSON report: {results['json_report']}")
        return

    # Multi-dataset mode: use first as primary, rest as additional
    items = list(discovered.items())
    primary_label, primary_path = items[0]
    additional_dirs = {label: path for label, path in items[1:]}

    output_dir = args.output_dir or base_path

    print(f"\nGenerating combined report...")
    print(f"  Primary dataset: {primary_label}")
    for label in additional_dirs:
        print(f"  Additional dataset: {label}")

    results = analyze_run(
        args.run_dir,
        output_dir,
        include_html=not args.no_html,
        include_json=not args.no_json,
        compare_dir=primary_path,
        additional_compare_dirs=additional_dirs,
    )
    print(f"\nAnalyzed {results['items']} items, {results['failures']} failures")
    if "html_report" in results:
        print(f"Wrote HTML report: {results['html_report']}")
    if "json_report" in results:
        print(f"Wrote JSON report (combined): {results['json_report']}")
    if "dataset_json_reports" in results:
        print("Wrote individual dataset JSON reports:")
        for dataset_key, json_path in results["dataset_json_reports"].items():
            print(f"  - {dataset_key}: {json_path}")


if __name__ == "__main__":
    main()
