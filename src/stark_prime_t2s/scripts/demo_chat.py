"""Interactive demo chat with the STaRK-Prime agent.

This module provides the entry point for the demo-chat CLI command.
"""

import argparse
import sys

from stark_prime_t2s.agent.agent import (
    create_stark_prime_agent,
    create_stark_prime_entity_resolver_agent,
    create_stark_prime_sparql_agent,
    create_stark_prime_sql_agent,
    run_agent_query_sync,
)
from stark_prime_t2s.config import OPENAI_MODEL


WELCOME_MESSAGE = """
╔══════════════════════════════════════════════════════════════╗
║           STaRK-Prime Text-to-SQL/SPARQL Agent               ║
╠══════════════════════════════════════════════════════════════╣
║  Ask questions about diseases, drugs, genes, proteins,       ║
║  pathways, and their relationships in the STaRK-Prime        ║
║  biomedical knowledge base.                                  ║
║                                                              ║
║  The agent will automatically choose SQL or SPARQL to        ║
║  query the knowledge base and provide answers.               ║
║                                                              ║
║  Commands:                                                   ║
║    - Type your question and press Enter                      ║
║    - Type 'quit' or 'exit' to exit                           ║
║    - Type 'help' for example questions                       ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLE_QUESTIONS = """
Example questions you can ask:

1. Simple lookups:
   - "How many diseases are in the knowledge base?"
   - "List 5 drugs with their names"
   - "What types of entities exist?"

2. Relationship queries:
   - "What drugs are indicated for treating diabetes?"
   - "Find genes associated with breast cancer"
   - "What are the side effects of aspirin?"

3. Complex queries:
   - "Find genes that are associated with both Alzheimer's disease and Parkinson's disease"
   - "What pathways involve the BRCA1 gene?"
   - "Which drugs have contraindications with heart disease?"

4. From the STaRK-Prime benchmark:
   - "I am looking for a gene or protein that plays a role in ribosomal operations"
   - "What is the name of the inflammatory disease that primarily targets the small intestine?"
"""


def check_services_ready() -> bool:
    """Check if the Docker services are ready."""
    from stark_prime_t2s.tools.execute_query import get_sql_store, get_sparql_store
    from stark_prime_t2s.tools.entity_resolver import build_entity_index

    print("  ✓ Checking Docker services...")

    # Check PostgreSQL
    try:
        store = get_sql_store()
        if not store.is_available():
            print("ERROR: PostgreSQL has no data!")
            print("Run: python scripts/build_prime_stores.py")
            return False
        print("  ✓ PostgreSQL ready")
    except Exception as e:
        print(f"ERROR: PostgreSQL unavailable: {e}")
        print("Make sure Docker is running: docker-compose up -d")
        return False

    # Check Fuseki
    try:
        store = get_sparql_store()
        if not store.is_available():
            print("WARNING: Fuseki has no data (SPARQL queries may fail)")
        else:
            print("  ✓ Fuseki ready")
    except Exception as e:
        print(f"WARNING: Fuseki unavailable: {e}")

    # Check Qdrant
    try:
        count = build_entity_index()
        if count == 0:
            print("WARNING: Qdrant has no data (entity search may fail)")
            print("Run: python scripts/build_prime_stores.py")
        else:
            print(f"  ✓ Qdrant ready ({count} entities)")
    except Exception as e:
        print(f"WARNING: Qdrant unavailable: {e}")

    return True


def main():
    """Main entry point for demo-chat command."""
    parser = argparse.ArgumentParser(description="Interactive demo chat with STaRK-Prime agent")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model to use (default: {OPENAI_MODEL})",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=["auto", "sql", "sparql", "entity"],
        help="Agent mode: auto (SQL+SPARQL), sql-only, sparql-only, or entity-only",
    )

    args = parser.parse_args()

    # Check services
    if not check_services_ready():
        sys.exit(1)

    print(WELCOME_MESSAGE)

    # Select agent mode
    if args.agent:
        agent_mode = args.agent
    else:
        prompt = "Select agent mode [auto/sql/sparql/entity] (default: auto): "
        choice = input(prompt).strip().lower()
        agent_mode = choice or "auto"
        if agent_mode not in ("auto", "sql", "sparql", "entity"):
            print("Invalid selection. Using auto mode.")
            agent_mode = "auto"

    # Create agent
    print("  ✓ Initializing agent...")
    try:
        if agent_mode == "sql":
            agent = create_stark_prime_sql_agent(model=args.model)
        elif agent_mode == "sparql":
            agent = create_stark_prime_sparql_agent(model=args.model)
        elif agent_mode == "entity":
            agent = create_stark_prime_entity_resolver_agent(model=args.model)
        else:
            agent = create_stark_prime_agent(model=args.model)
    except Exception as e:
        print(f"ERROR: Failed to create agent: {e}")
        print()
        print("Make sure you have set OPENAI_API_KEY in your .env file or environment.")
        sys.exit(1)

    print()
    print("-" * 60)
    print()

    # Chat loop
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if question.lower() == "help":
            print(EXAMPLE_QUESTIONS)
            continue

        print()
        print("Agent: Thinking...")

        try:
            result = run_agent_query_sync(agent, question)
            print()
            # Display reasoning
            print(f"Agent Reasoning: {result['reasoning']}")
            print()
            # Display the IDs (answer)
            print(f"Agent Answer (IDs): {result['node_ids']}")

            # Show tool calls if any
            if result["tool_calls"]:
                print()
                print(f"  [Made {len(result['tool_calls'])} query(ies)]")

        except Exception as e:
            print(f"ERROR: {e}")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
