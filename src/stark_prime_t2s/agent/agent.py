"""LangChain v1 agent for STaRK-Prime text-to-SQL/SPARQL with Langfuse v3 observability."""

from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from stark_prime_t2s.config import (
    LANGFUSE_BASE_URL,
    LANGFUSE_ENABLED,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LLM_PROVIDER,
    MAX_AGENT_ITERATIONS,
    MAX_QUERY_ROWS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from stark_prime_t2s.tools.entity_resolver import (
    build_entity_index,
    get_search_entities_tool,
)
from stark_prime_t2s.tools.execute_query import (
    get_execute_query_tool,
    get_execute_sparql_query_tool,
    get_execute_sql_query_tool,
    get_schema_and_vocab_summary,
    get_sparql_vocab_summary,
    get_sql_schema_summary,
)


SYSTEM_PROMPT_TEMPLATE = """You are an expert biomedical knowledge base analyst. Your task is to answer questions about diseases, drugs, genes/proteins, pathways, molecular functions, and their relationships using the STaRK-Prime knowledge base.

## Available Tools

You have access to TWO tools:

1. **search_entities_tool** - Semantic search to find entities by name/description
   - Use this FIRST to resolve entity names to their IDs
   - Handles synonyms, partial matches, and related terms
   - Returns entity IDs that you can use in queries

2. **execute_query_tool** - Execute SQL or SPARQL queries
    - Use AFTER finding entity IDs with search_entities_tool
    - Supports SQL (PostgreSQL) and SPARQL (Fuseki)

## Two-Stage Query Process (IMPORTANT!)

**ALWAYS follow this two-stage approach:**

### Stage 1: Entity Resolution
When a question mentions specific entities (diseases, drugs, genes, etc.):
1. Use `search_entities_tool` to find the entity IDs
2. Note the returned IDs for use in your query

Example: For "What genes are associated with both diabetes and hypertension?"
→ First: Call BOTH searches in parallel:
  - `search_entities_tool("diabetes", "disease")`
  - `search_entities_tool("hypertension", "disease")`
→ Get: diabetes ID 12345, hypertension ID 67890

### Stage 2: Query Execution  
Use the resolved entity IDs in your SQL or SPARQL query:
→ Then: `execute_query_tool("sql", "SELECT ...")` or `execute_query_tool("sparql", "PREFIX sp: ...")`

## Exploration Strategy

Adapt your approach based on the query structure:

### Strategy 1: Queries without explicit entity mentions
When the query does not explicitly mention specific entities (product names, paper titles, gene names, author names, etc.):
1. Use `search_entities_tool` with the full question as the query and `top_k=15` to cast a wide net.

### Strategy 2: Queries with explicit entity mentions
When specific entities are mentioned:
1. Resolve each entity with `search_entities_tool` (use `entity_type` when obvious).
2. Proceed to query execution with the resolved IDs.

### Strategy 3: Multi-entity or complex queries
For queries involving multiple entities or combined criteria:
1. Disambiguate all mentioned entities.
2. Explore neighborhoods of key entities with relevant filters.
3. Combine information from multiple exploration paths.

## Query Language Selection

Choose the appropriate query language based on the question:

- **SQL** is better for:
  - Simple lookups by ID, name, or type
  - Aggregations (COUNT, AVG, MAX, etc.)
  - Joins between multiple entity types
  - Filtering with complex conditions
  - Questions asking "how many" or "list all"

- **SPARQL** is better for:
  - Path traversal and relationship exploration
  - Finding connections between entities
  - Pattern matching across the knowledge graph
  - Questions like "what is connected to X through Y"

## Knowledge Base Schema

{schema_and_vocab}

## Query Guidelines

1. **Entity Resolution First**: ALWAYS use search_entities_tool to find entity IDs before querying.
   Do NOT try to match entity names with SQL LIKE or SPARQL FILTER - use semantic search instead.

2. **Read-only only**: SQL must be SELECT-only; SPARQL must be read-only (SELECT, ASK, CONSTRUCT, DESCRIBE). No INSERT/UPDATE/DELETE.

3. **Limit results**: Always use LIMIT {max_rows} unless the user specifically asks for more.

4. **Be precise**: Use exact table/column names from the schema above.

5. **Handle errors**: If a query fails, analyze the error and try a corrected query.

6. **Node IDs are answers**: When the question asks "which" or "what", the answer is typically node IDs.
   The benchmark expects node IDs (integers) as answers, not names.

## Answer Format

You MUST output ONLY a JSON object with exactly two fields. No markdown, no code blocks, no explanatory text before or after.

CORRECT output:
{"ids": [123, 456, 789], "reasoning": "Found 3 genes associated with diabetes through indication relationships"}

INCORRECT outputs (DO NOT DO THESE):
- ```json\n{{"ids": [123], "reasoning": "..."}}\n```
- The answer is: {{"ids": [123], "reasoning": "..."}}
- Based on my analysis... {{"ids": [123], "reasoning": "..."}}
- "ids": [123], "reasoning": "..."

**REQUIREMENTS:**
- Output ONLY the raw JSON object starting with { and ending with }
- The `ids` field must be an array of integers, empty array [] if no results
- The `reasoning` field must be a string explaining your process
- IDs preserve ranking order for Hit@1 and MRR metrics

## CRITICAL: Efficiency Guidelines

### Parallelization
- When resolving multiple entities, call ALL searches IN PARALLEL in ONE turn (max 4-5 searches)

### Retry Limits
- Query errors: max 2 corrective attempts per query
- Zero-row results: max 2 query reformulations (broaden filters, try alternative relationships)
- After limits exhausted, report partial results or fallback message

### Answer Strategy
- If a query returns 0 rows, adjust and retry (within retry limits above)
- If still empty, report: "Based on entity resolution, I found X, Y, Z entities relevant to your question.
  However, the knowledge base does not contain direct relationships connecting all these criteria."
- Partial results are valuable - report entity IDs even without relationship data

## Query Examples (SQL + SPARQL)

Example SQL workflow:
1. `search_entities_tool("breast cancer", "disease")` → ID: 789
2. `execute_query_tool("sql", "SELECT dst_id FROM indication WHERE src_id = 789")`

Example SPARQL workflow:
1. `search_entities_tool("insulin", "gene_protein")` → ID: 456
2. `execute_query_tool("sparql", "PREFIX sp: <http://stark.stanford.edu/prime/> SELECT ?related WHERE {{ <http://stark.stanford.edu/prime/node/456> sp:associatedWith ?related }} LIMIT 5")`

Now answer the user's question using the two-stage approach.
"""


SQL_ONLY_SYSTEM_PROMPT_TEMPLATE = """You are an expert biomedical knowledge base analyst. Your task is to answer questions about diseases, drugs, genes/proteins, pathways, molecular functions, and their relationships using the STaRK-Prime knowledge base.

## Available Tools

You have access to TWO tools:

1. **search_entities_tool** - Semantic search to find entities by name/description
   - Use this FIRST to resolve entity names to their IDs
   - Handles synonyms, partial matches, and related terms
   - Returns entity IDs that you can use in queries

2. **execute_sql_query_tool** - Execute SQL queries only
   - Use AFTER finding entity IDs with search_entities_tool
   - Supports SQL (PostgreSQL) only

## Two-Stage Query Process (IMPORTANT!)

**ALWAYS follow this two-stage approach:**

### Stage 1: Entity Resolution
When a question mentions specific entities (diseases, drugs, genes, etc.):
1. Use `search_entities_tool` to find the entity IDs
2. Note the returned IDs for use in your query

Example: For "What genes are associated with both diabetes and hypertension?"
→ First: Call BOTH searches in parallel:
  - `search_entities_tool("diabetes", "disease")`
  - `search_entities_tool("hypertension", "disease")`
→ Get: diabetes ID 12345, hypertension ID 67890

### Stage 2: Query Execution
Use the resolved entity IDs in your SQL query:
→ Then: `execute_sql_query_tool("SELECT ... WHERE ... 12345 ... 67890 ...")`

## Exploration Strategy

Adapt your approach based on the query structure:

### Strategy 1: Queries without explicit entity mentions
When the query does not explicitly mention specific entities (product names, paper titles, gene names, author names, etc.):
1. Use `search_entities_tool` with the full question as the query and `top_k=15` to cast a wide net.

### Strategy 2: Queries with explicit entity mentions
When specific entities are mentioned:
1. Resolve each entity with `search_entities_tool` (use `entity_type` when obvious).
2. Proceed to query execution with the resolved IDs.

### Strategy 3: Multi-entity or complex queries
For queries involving multiple entities or combined criteria:
1. Disambiguate all mentioned entities.
2. Explore neighborhoods of key entities with relevant filters.
3. Combine information from multiple exploration paths.

## Query Language Selection

You MUST use SQL only.

## Knowledge Base Schema

{schema_and_vocab}

## Query Guidelines

1. **Entity Resolution First**: ALWAYS use search_entities_tool to find entity IDs before querying.
   Do NOT try to match entity names with SQL LIKE - use semantic search instead.

2. **Read-only only**: Only SELECT queries are allowed. No INSERT, UPDATE, DELETE, etc.

3. **Limit results**: Always use LIMIT {max_rows} unless the user specifically asks for more.

4. **Be precise**: Use exact table/column names from the schema above.

5. **Handle errors**: If a query fails, analyze the error and try a corrected query.

6. **Node IDs are answers**: When the question asks "which" or "what", the answer is typically node IDs.
   The benchmark expects node IDs (integers) as answers, not names.

## Answer Format

You MUST output ONLY a JSON object with exactly two fields. No markdown, no code blocks, no explanatory text before or after.

CORRECT output:
{"ids": [123, 456, 789], "reasoning": "Found 3 genes associated with diabetes through indication relationships"}

INCORRECT outputs (DO NOT DO THESE):
- ```json\n{{"ids": [123], "reasoning": "..."}}\n```
- The answer is: {{"ids": [123], "reasoning": "..."}}
- Based on my analysis... {{"ids": [123], "reasoning": "..."}}
- "ids": [123], "reasoning": "..."

**REQUIREMENTS:**
- Output ONLY the raw JSON object starting with { and ending with }
- The `ids` field must be an array of integers, empty array [] if no results
- The `reasoning` field must be a string explaining your process
- IDs preserve ranking order for Hit@1 and MRR metrics

## CRITICAL: Efficiency Guidelines

### Parallelization
- When resolving multiple entities, call ALL searches IN PARALLEL in ONE turn (max 4-5 searches)

### Retry Limits
- Query errors: max 2 corrective attempts per query
- Zero-row results: max 2 query reformulations (broaden filters, try alternative relationships)
- After limits exhausted, report partial results or fallback message

### Answer Strategy
- If a query returns 0 rows, adjust and retry (within retry limits above)
- If still empty, report: "Based on entity resolution, I found X, Y, Z entities relevant to your question.
  However, the knowledge base does not contain direct relationships connecting all these criteria."
- Partial results are valuable - report entity IDs even without relationship data

Example workflow:
1. `search_entities_tool("breast cancer", "disease")` → ID: 789
2. `execute_sql_query_tool("SELECT dst_id FROM indication WHERE src_id = 789")`

Now answer the user's question using the two-stage approach.
"""


SPARQL_ONLY_SYSTEM_PROMPT_TEMPLATE = """You are an expert biomedical knowledge base analyst. Your task is to answer questions about diseases, drugs, genes/proteins, pathways, molecular functions, and their relationships using the STaRK-Prime knowledge base.

## Available Tools

You have access to TWO tools:

1. **search_entities_tool** - Semantic search to find entities by name/description
   - Use this FIRST to resolve entity names to their IDs
   - Handles synonyms, partial matches, and related terms
   - Returns entity IDs that you can use in queries

2. **execute_sparql_query_tool** - Execute SPARQL queries only
   - Use AFTER finding entity IDs with search_entities_tool
   - Supports SPARQL (Fuseki) only

## Two-Stage Query Process (IMPORTANT!)

**ALWAYS follow this two-stage approach:**

### Stage 1: Entity Resolution
When a question mentions specific entities (diseases, drugs, genes, etc.):
1. Use `search_entities_tool` to find the entity IDs
2. Note the returned IDs for use in your query

Example: For "What genes are associated with both diabetes and hypertension?"
→ First: Call BOTH searches in parallel:
  - `search_entities_tool("diabetes", "disease")`
  - `search_entities_tool("hypertension", "disease")`
→ Get: diabetes ID 12345, hypertension ID 67890

### Stage 2: Query Execution
Use the resolved entity IDs in your SPARQL query:
→ Then: `execute_sparql_query_tool("PREFIX sp: <http://stark.stanford.edu/prime/> SELECT ...")`

## Exploration Strategy

Adapt your approach based on the query structure:

### Strategy 1: Queries without explicit entity mentions
When the query does not explicitly mention specific entities (product names, paper titles, gene names, author names, etc.):
1. Use `search_entities_tool` with the full question as the query and `top_k=15` to cast a wide net.

### Strategy 2: Queries with explicit entity mentions
When specific entities are mentioned:
1. Resolve each entity with `search_entities_tool` (use `entity_type` when obvious).
2. Proceed to query execution with the resolved IDs.

### Strategy 3: Multi-entity or complex queries
For queries involving multiple entities or combined criteria:
1. Disambiguate all mentioned entities.
2. Explore neighborhoods of key entities with relevant filters.
3. Combine information from multiple exploration paths.

## Query Language Selection

You MUST use SPARQL only.

## Knowledge Base Schema

{schema_and_vocab}

## Query Guidelines

1. **Entity Resolution First**: ALWAYS use search_entities_tool to find entity IDs before querying.
   Do NOT try to match entity names with SPARQL FILTER - use semantic search instead.

2. **Read-only only**: Only SELECT, ASK, CONSTRUCT, DESCRIBE allowed. Do not use UPDATE/INSERT/DELETE.

3. **Limit results**: Always use LIMIT {max_rows} unless the user specifically asks for more.

4. **Be precise**: Use exact classes/properties from the vocabulary above.

5. **Handle errors**: If a query fails, analyze the error and try a corrected query.

6. **Node IDs are answers**: When the question asks "which" or "what", the answer is typically node IDs.
   The benchmark expects node IDs (integers) as answers, not names.

## Answer Format

You MUST output ONLY a JSON object with exactly two fields. No markdown, no code blocks, no explanatory text before or after.

CORRECT output:
{{"ids": [123, 456, 789], "reasoning": "Found 3 genes associated with diabetes through indication relationships"}}

INCORRECT outputs (DO NOT DO THESE):
- ```json\n{{"ids": [123], "reasoning": "..."}}\n```
- The answer is: {{"ids": [123], "reasoning": "..."}}
- Based on my analysis... {{"ids": [123], "reasoning": "..."}}
- "ids": [123], "reasoning": "..."

**REQUIREMENTS:**
- Output ONLY the raw JSON object starting with {{ and ending with }}
- The `ids` field must be an array of integers, empty array [] if no results
- The `reasoning` field must be a string explaining your process
- IDs preserve ranking order for Hit@1 and MRR metrics

## CRITICAL: Efficiency Guidelines

### Parallelization
- When resolving multiple entities, call ALL searches IN PARALLEL in ONE turn (max 4-5 searches)

### Retry Limits
- Query errors: max 2 corrective attempts per query
- Zero-row results: max 2 query reformulations (broaden filters, try alternative relationships)
- After limits exhausted, report partial results or fallback message

### Answer Strategy
- If a query returns 0 rows, adjust and retry (within retry limits above)
- If still empty, report: "Based on entity resolution, I found X, Y, Z entities relevant to your question.
  However, the knowledge base does not contain direct relationships connecting all these criteria."
- Partial results are valuable - report entity IDs even without relationship data

Example SPARQL workflow:
1. `search_entities_tool("insulin", "gene_protein")` → ID: 456
2. `execute_sparql_query_tool("PREFIX sp: <http://stark.stanford.edu/prime/> SELECT ?related WHERE {{ sp:node/456 sp:associatedWith ?related }} LIMIT 5")`

Now answer the user's question using the two-stage approach.
"""


ENTITY_ONLY_SYSTEM_PROMPT_TEMPLATE = """You are an expert biomedical knowledge base analyst. Your task is to resolve entity mentions to STaRK-Prime node IDs using semantic search.

## Available Tools

You have access to ONE tool:

1. **search_entities_tool** - Semantic search to find entities by name/description
   - Use this to resolve entity names to their IDs
   - Handles synonyms, partial matches, and related terms

## Entity Resolution Strategy (IMPORTANT!)

### Strategy 1: Queries with explicit entity mentions
When specific entities are mentioned:
1. Resolve each entity with `search_entities_tool` (use `entity_type` when obvious)
2. Collect the top results and return their IDs

### Strategy 2: Queries without explicit entity mentions
When the query does not explicitly mention specific entities:
1. Use `search_entities_tool` with the full question as the query and `top_k=15`
2. Return IDs from the best-matching results

## Output Rules

You MUST output ONLY a JSON object with exactly two fields. No markdown, no code blocks, no explanatory text before or after.

CORRECT output:
{"ids": [123, 456, 789], "reasoning": "Resolved entities using semantic search and returned the top matching IDs"}

INCORRECT outputs (DO NOT DO THESE):
- ```json\n{{"ids": [123], "reasoning": "..."}}\n```
- The answer is: {{"ids": [123], "reasoning": "..."}}
- Based on my analysis... {{"ids": [123], "reasoning": "..."}}
- "ids": [123], "reasoning": "..."

**REQUIREMENTS:**
- Output ONLY the raw JSON object starting with { and ending with }
- The `ids` field must be an array of integers, empty array [] if no results
- The `reasoning` field must be a string explaining your process
- IDs preserve ranking order for Hit@1 and MRR metrics

## CRITICAL: Efficiency Guidelines

### Parallelization
- When resolving multiple entities, call ALL searches IN PARALLEL in ONE turn (max 4-5 searches)

Now answer the user's question using the entity resolution strategy.
"""


# ---------------------------------------------------------------------------
# Langfuse v3 Observability
# ---------------------------------------------------------------------------

_langfuse_initialized = False
_langfuse_handler = None


def _init_langfuse():
    """Initialize Langfuse client (singleton pattern for v3).

    In v3, the Langfuse client uses a singleton pattern. We initialize it once
    with environment variables and then access it via get_client().
    """
    global _langfuse_initialized

    if _langfuse_initialized:
        return True

    if not LANGFUSE_ENABLED:
        return False

    if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
        print("  Warning: Langfuse credentials not set, tracing disabled")
        return False

    try:
        import os

        # Set environment variables for Langfuse v3
        os.environ["LANGFUSE_SECRET_KEY"] = LANGFUSE_SECRET_KEY
        os.environ["LANGFUSE_PUBLIC_KEY"] = LANGFUSE_PUBLIC_KEY
        os.environ["LANGFUSE_BASE_URL"] = LANGFUSE_BASE_URL

        # Import and initialize the Langfuse client (v3 singleton pattern)
        from langfuse import Langfuse

        # Initialize the singleton client
        Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_BASE_URL,
        )

        _langfuse_initialized = True
        print(f"  ✓ Langfuse v3 tracing enabled ({LANGFUSE_BASE_URL})", flush=True)
        return True

    except ImportError:
        print("  Warning: langfuse not installed, tracing disabled")
        return False
    except Exception as e:
        print(f"  Warning: Could not initialize Langfuse: {e}")
        return False


def get_langfuse_handler():
    """Get the Langfuse callback handler for LangChain tracing.

    In v3, CallbackHandler() takes no constructor arguments.
    Trace attributes (user_id, session_id, tags) are passed via
    chain config metadata with 'langfuse_' prefix.

    Returns:
        CallbackHandler if Langfuse is configured, None otherwise.
    """
    global _langfuse_handler

    if not _init_langfuse():
        return None

    if _langfuse_handler is None:
        try:
            from langfuse.langchain import CallbackHandler

            # v3: CallbackHandler takes no constructor arguments
            _langfuse_handler = CallbackHandler()
        except Exception as e:
            print(f"  Warning: Could not create Langfuse handler: {e}")
            return None

    return _langfuse_handler


def _parse_structured_output(answer: str) -> dict[str, Any]:
    """Parse structured JSON output from the agent.

    Args:
        answer: The raw answer text from the agent

    Returns:
        Dict with 'ids' (list of ints) and 'reasoning' (str) keys

    Raises:
        ValueError: If the output is not valid JSON with required fields
    """
    import json
    import re

    cleaned = answer.strip()

    # Remove markdown code block wrapper if present
    if cleaned.startswith("```"):
        # Extract content from ```json ... ``` or ``` ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1)

    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

    ids = parsed.get("ids")
    reasoning = parsed.get("reasoning")

    if ids is None or reasoning is None:
        raise ValueError("JSON must contain 'ids' and 'reasoning' fields")

    if not isinstance(ids, list):
        raise ValueError(f"'ids' must be a list, got {type(ids).__name__}")

    # Convert to integers
    try:
        ids = [int(x) for x in ids]
    except (ValueError, TypeError) as e:
        raise ValueError(f"All 'ids' must be integers: {e}")

    return {"ids": ids, "reasoning": reasoning}


def get_callbacks() -> list:
    """Get the list of callback handlers for the agent.

    Returns:
        List of callback handlers (may be empty if none configured).
    """
    callbacks = []

    langfuse = get_langfuse_handler()
    if langfuse:
        callbacks.append(langfuse)

    return callbacks


def flush_langfuse():
    """Flush any pending Langfuse events.

    In v3, flush is called on the client instance via get_client().
    Call this in short-lived scripts to ensure all events are sent.
    """
    if not _langfuse_initialized:
        return

    try:
        from langfuse import get_client

        client = get_client()
        client.flush()
    except Exception:
        pass  # Silently ignore flush errors


# ---------------------------------------------------------------------------
# Agent Creation
# ---------------------------------------------------------------------------


def get_system_prompt() -> str:
    """Generate the system prompt with current schema/vocabulary.

    Returns:
        The complete system prompt string
    """
    schema_and_vocab = get_schema_and_vocab_summary()
    return SYSTEM_PROMPT_TEMPLATE.format(
        schema_and_vocab=schema_and_vocab,
        max_rows=MAX_QUERY_ROWS,
    )


def get_sql_only_system_prompt() -> str:
    """Generate the SQL-only system prompt with current schema/vocabulary."""
    schema_and_vocab = get_sql_schema_summary()
    return SQL_ONLY_SYSTEM_PROMPT_TEMPLATE.format(
        schema_and_vocab=schema_and_vocab,
        max_rows=MAX_QUERY_ROWS,
    )


def get_sparql_only_system_prompt() -> str:
    """Generate the SPARQL-only system prompt with current schema/vocabulary."""
    schema_and_vocab = get_sparql_vocab_summary()
    return SPARQL_ONLY_SYSTEM_PROMPT_TEMPLATE.format(
        schema_and_vocab=schema_and_vocab,
        max_rows=MAX_QUERY_ROWS,
    )


def get_entity_only_system_prompt() -> str:
    """Generate the entity-only system prompt."""
    return ENTITY_ONLY_SYSTEM_PROMPT_TEMPLATE


def create_stark_prime_agent(
    model: str | None = None,
    temperature: float = 0.0,
    build_entity_index_on_start: bool = True,
    **kwargs: Any,
):
    """Create a LangChain agent for STaRK-Prime queries.

    This creates an agent using LangChain v1's create_agent function,
    which internally uses LangGraph for the agent loop.

    The agent uses a two-stage approach:
    1. Entity resolution via semantic search (search_entities_tool)
    2. Query execution via SQL/SPARQL (execute_query_tool)

    Args:
        model: The model to use. Defaults to config.OPENAI_MODEL or config.OPENROUTER_MODEL.
        temperature: The temperature for the model. Defaults to 0.0 for deterministic output.
        build_entity_index_on_start: Whether to build/load the entity index on startup.
            Set to False if you've already built the index. Defaults to True.
        **kwargs: Additional arguments passed to init_chat_model.
            Use provider="openai" or provider="openrouter" to override the default.

    Returns:
        A LangChain agent that can answer questions about STaRK-Prime.
    """
    # Determine which provider to use and validate configuration
    provider = kwargs.pop("provider", None) or LLM_PROVIDER

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENROUTER_API_KEY
        base_url = OPENROUTER_BASE_URL
        model_name = model or OPENROUTER_MODEL
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENAI_API_KEY
        base_url = None
        model_name = model or OPENAI_MODEL
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'openrouter'.")

    # Build/load entity index for semantic search
    if build_entity_index_on_start:
        print("Building entity index for semantic search...")
        build_entity_index()

    # Initialize the chat model
    init_kwargs = {
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    if "model_provider" not in kwargs:
        init_kwargs["model_provider"] = "openai"
    init_kwargs.update(kwargs)

    llm = init_chat_model(
        model_name,
        **init_kwargs,
    )

    print(f"  Agent ready (provider: {provider}, model: {model_name})")

    # Get the tools (two-stage approach)
    search_tool = get_search_entities_tool()
    query_tool = get_execute_query_tool()

    # Get the system prompt
    system_prompt = get_system_prompt()

    # Create the agent using LangChain v1's create_agent
    agent = create_agent(
        llm,
        tools=[search_tool, query_tool],
        system_prompt=system_prompt,
    )

    return agent


def create_stark_prime_sql_agent(
    model: str | None = None,
    temperature: float = 0.0,
    build_entity_index_on_start: bool = True,
    **kwargs: Any,
):
    """Create a LangChain agent that can only execute SQL queries."""
    provider = kwargs.pop("provider", None) or LLM_PROVIDER

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENROUTER_API_KEY
        base_url = OPENROUTER_BASE_URL
        model_name = model or OPENROUTER_MODEL
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENAI_API_KEY
        base_url = None
        model_name = model or OPENAI_MODEL
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'openrouter'.")

    if build_entity_index_on_start:
        print("Building entity index for semantic search...")
        build_entity_index()

    init_kwargs = {
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    if "model_provider" not in kwargs:
        init_kwargs["model_provider"] = "openai"
    init_kwargs.update(kwargs)

    llm = init_chat_model(
        model_name,
        **init_kwargs,
    )

    print(f"  Agent ready (provider: {provider}, model: {model_name})")

    search_tool = get_search_entities_tool()
    query_tool = get_execute_sql_query_tool()
    system_prompt = get_sql_only_system_prompt()

    agent = create_agent(
        llm,
        tools=[search_tool, query_tool],
        system_prompt=system_prompt,
    )

    return agent


def create_stark_prime_sparql_agent(
    model: str | None = None,
    temperature: float = 0.0,
    build_entity_index_on_start: bool = True,
    **kwargs: Any,
):
    """Create a LangChain agent that can only execute SPARQL queries."""
    provider = kwargs.pop("provider", None) or LLM_PROVIDER

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENROUTER_API_KEY
        base_url = OPENROUTER_BASE_URL
        model_name = model or OPENROUTER_MODEL
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENAI_API_KEY
        base_url = None
        model_name = model or OPENAI_MODEL
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'openrouter'.")

    if build_entity_index_on_start:
        print("Building entity index for semantic search...")
        build_entity_index()

    init_kwargs = {
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    if "model_provider" not in kwargs:
        init_kwargs["model_provider"] = "openai"
    init_kwargs.update(kwargs)

    llm = init_chat_model(
        model_name,
        **init_kwargs,
    )

    print(f"  Agent ready (provider: {provider}, model: {model_name})")

    search_tool = get_search_entities_tool()
    query_tool = get_execute_sparql_query_tool()
    system_prompt = get_sparql_only_system_prompt()

    agent = create_agent(
        llm,
        tools=[search_tool, query_tool],
        system_prompt=system_prompt,
    )

    return agent


def create_stark_prime_entity_resolver_agent(
    model: str | None = None,
    temperature: float = 0.0,
    build_entity_index_on_start: bool = True,
    **kwargs: Any,
):
    """Create a LangChain agent that can only resolve entities via semantic search."""
    provider = kwargs.pop("provider", None) or LLM_PROVIDER

    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENROUTER_API_KEY
        base_url = OPENROUTER_BASE_URL
        model_name = model or OPENROUTER_MODEL
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it in your .env file or environment."
            )
        api_key = OPENAI_API_KEY
        base_url = None
        model_name = model or OPENAI_MODEL
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Use 'openai' or 'openrouter'.")

    if build_entity_index_on_start:
        print("Building entity index for semantic search...")
        build_entity_index()

    init_kwargs = {
        "temperature": temperature,
        "api_key": api_key,
    }
    if base_url:
        init_kwargs["base_url"] = base_url
    if "model_provider" not in kwargs:
        init_kwargs["model_provider"] = "openai"
    init_kwargs.update(kwargs)

    llm = init_chat_model(
        model_name,
        **init_kwargs,
    )

    print(f"  Agent ready (provider: {provider}, model: {model_name})")

    search_tool = get_search_entities_tool()
    system_prompt = get_entity_only_system_prompt()

    agent = create_agent(
        llm,
        tools=[search_tool],
        system_prompt=system_prompt,
    )

    return agent


# ---------------------------------------------------------------------------
# Agent Invocation with Tracing
# ---------------------------------------------------------------------------


async def run_agent_query(
    agent,
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Run a query through the agent and extract the answer.

    Args:
        agent: The LangChain agent
        question: The user's question
        session_id: Optional session ID for Langfuse tracing
        user_id: Optional user ID for Langfuse tracing
        trace_name: Optional trace name for Langfuse
        tags: Optional tags for Langfuse tracing

    Returns:
        Dict with 'node_ids', 'reasoning', 'messages', and 'tool_calls' keys
    """
    # Build config with callbacks
    callbacks = get_callbacks()
    config: dict[str, Any] = {}

    if callbacks:
        config["callbacks"] = callbacks
        config["run_name"] = trace_name or "stark_prime_query"

        # v3: Trace attributes go in metadata with 'langfuse_' prefix
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if tags:
            metadata["langfuse_tags"] = tags

        if metadata:
            config["metadata"] = metadata

    # Set recursion limit to prevent runaway loops
    config["recursion_limit"] = MAX_AGENT_ITERATIONS

    def _extract_final_answer(result: dict[str, Any]) -> tuple[str, list, list]:
        messages = result.get("messages", [])
        final_answer = ""
        tool_calls = []

        for msg in messages:
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "ai":
                    final_answer = msg.content
                elif msg.type == "tool":
                    tool_calls.append(
                        {
                            "name": getattr(msg, "name", "unknown"),
                            "content": (
                                msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                            ),
                        }
                    )

        return final_answer, messages, tool_calls

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        final_answer, messages, tool_calls = _extract_final_answer(result)

        try:
            structured = _parse_structured_output(final_answer)
            return {
                "node_ids": structured["ids"],
                "reasoning": structured["reasoning"],
                "messages": messages,
                "tool_calls": tool_calls,
            }
        except ValueError:
            if attempt == max_attempts:
                raise
            continue

    raise RuntimeError("Failed to parse agent output after retries")


def run_agent_query_sync(
    agent,
    question: str,
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Synchronous version of run_agent_query.

    Args:
        agent: The LangChain agent
        question: The user's question
        session_id: Optional session ID for Langfuse tracing
        user_id: Optional user ID for Langfuse tracing
        trace_name: Optional trace name for Langfuse
        tags: Optional tags for Langfuse tracing

    Returns:
        Dict with 'node_ids', 'reasoning', 'messages', and 'tool_calls' keys
    """
    # Build config with callbacks
    callbacks = get_callbacks()
    config: dict[str, Any] = {}

    if callbacks:
        config["callbacks"] = callbacks
        config["run_name"] = trace_name or "stark_prime_query"

        # v3: Trace attributes go in metadata with 'langfuse_' prefix
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if user_id:
            metadata["langfuse_user_id"] = user_id
        if tags:
            metadata["langfuse_tags"] = tags

        if metadata:
            config["metadata"] = metadata

    # Set recursion limit to prevent runaway loops
    config["recursion_limit"] = MAX_AGENT_ITERATIONS

    def _extract_final_answer(result: dict[str, Any]) -> tuple[str, list, list]:
        messages = result.get("messages", [])
        final_answer = ""
        tool_calls = []

        for msg in messages:
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "ai":
                    final_answer = msg.content
                elif msg.type == "tool":
                    tool_calls.append(
                        {
                            "name": getattr(msg, "name", "unknown"),
                            "content": (
                                msg.content[:500] + "..." if len(msg.content) > 500 else msg.content
                            ),
                        }
                    )

        return final_answer, messages, tool_calls

    def _summarize_text(text: str | None, limit: int = 400) -> str:
        if not text:
            return "<empty>"
        cleaned = text.replace("\n", " ").strip()
        return cleaned if len(cleaned) <= limit else f"{cleaned[:limit]}..."

    def _summarize_messages(messages: list, limit: int = 3) -> list[str]:
        summaries: list[str] = []
        for msg in messages[-limit:]:
            msg_type = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", None)
            summaries.append(f"{msg_type}:{_summarize_text(content, 120)}")
        return summaries

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config=config,
        )

        final_answer, messages, tool_calls = _extract_final_answer(result)

        try:
            structured = _parse_structured_output(final_answer)
            return {
                "node_ids": structured["ids"],
                "reasoning": structured["reasoning"],
                "messages": messages,
                "tool_calls": tool_calls,
            }
        except ValueError as exc:
            if attempt == max_attempts:
                tool_names = [call.get("name", "unknown") for call in tool_calls][:3]
                summary = _summarize_text(final_answer)
                recent_messages = _summarize_messages(messages)
                raise ValueError(
                    "Failed to parse agent output. "
                    f"final_answer={summary} tool_calls={tool_names} recent_messages={recent_messages} error={exc}"
                ) from exc
            continue

    raise RuntimeError("Failed to parse agent output after retries")


if __name__ == "__main__":
    # Quick test
    print("Creating STaRK-Prime agent...")
    agent = create_stark_prime_agent()

    print("\nTesting with a sample question...")
    question = "How many diseases are in the knowledge base?"
    result = run_agent_query_sync(
        agent,
        question,
        trace_name="test_query",
        tags=["test", "stark-prime"],
    )

    print(f"\nQuestion: {question}")
    print(f"\nNode IDs: {result['node_ids']}")
    print(f"\nReasoning: {result['reasoning']}")
    print(f"\nTool calls: {len(result['tool_calls'])}")

    # Flush Langfuse events (important for short-lived scripts)
    flush_langfuse()
