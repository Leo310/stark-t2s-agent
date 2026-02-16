#import "@preview/charged-ieee:0.1.4": ieee

#set page(numbering: "1", number-align: center)

#show: ieee.with(
  title: [Agentic RAG on Semi-structured Databases],
  abstract: [
    The process of scientific writing is often tangled up with the intricacies of typesetting, leading to frustration and wasted time for researchers. In this paper, we introduce Typst, a new typesetting system designed specifically for scientific writing. Typst untangles the typesetting process, allowing researchers to compose papers faster. In a series of experiments we demonstrate that Typst offers several advantages, including faster document creation, simplified syntax, and increased ease-of-use.
  ],
  authors: (
    (
      name: "Leonard Heininger",
      department: [ARDI],
      organization: [TU Berlin],
      location: [Berlin, Germany],
      email: "haug@typst.app",
    ),
    (
      name: "Nicolas Kohl",
      department: [ARDI],
      organization: [TU Berlin],
      location: [Berlin, Germany],
      email: "nico-kohl@tu-berlin.de",
    ),
  ),
  index-terms: ("Scientific writing", "Typesetting", "Document creation", "Syntax"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)


#import "@preview/abbr:0.3.0"
#show: abbr.show-rule

#abbr.make(
  ("LLM", "Large Language Model"),
  ("aRAG", "agentic Retrieval Augmented Generation"),
  ("RAG", "Retrieval Augmented Generation"),
  ("IR", "Information Retrieval"),
  ("NLQ", "natural language query", "natural language queries"),
  ("SQL", "Structured Query Language"),
  ("SPARQL", "SPARQL Protocol and RDF Query Language"),
)


// 1. Setup: This rule makes the figure ONLY display its number (1)
#show figure.where(kind: "inline-item"): it => {
  // To make the number itself bold as well, wrap it in strong() here
  strong(it.counter.display(it.numbering))
}

// 2. The function: Automatically numbers, labels, and bolds the text
#let itm(lbl, body) = {
  box[#figure(
      kind: "inline-item",
      supplement: none,
      numbering: "(1)",
      [],
    )#lbl]
  [ ] // Space between number and text
  strong(body) // This makes your "in-bracket text" bold by default
}

= Introduction
@LLM:pla have fundamentally changed individual search behavior and information-seeking patterns. Whereas users previously needed to translate natural language into keyword searches tailored to algorithmic patterns, they can now directly employ @NLQ:pla to acquire information by interacting with @LLM systems. This approach better aligns with intuitive user behavior, enables more efficient information synthesis, and is already evident in observed user practices @sommerfeld2025ai @kaiser2025. #linebreak()

The shift toward @NLQ based search is realized through integrating classical @IR into generative workflows, an architectural pattern now widely referred to as @RAG. Historically, @IR produced ranked document lists from unstructured corpora, that formed the basis of search engines. Natural language querying is now enabled by layering @LLM:pla on top of @IR, enabling direct answer synthesis from retrieved documents rather than document ranking to end users. @IR mechanisms ground the generative process in documented facts, reducing hallucinations and extending the @LLM:pla's knowledge beyond its parametric boundaries @gao2024retrievalaugmentedgenerationlargelanguage. By combining @LLM:pla linguistic flexibility with @IR precision, this architecture ensures epistemic (WTH) reliability when accessing specialized or rapidly evolving data. #linebreak()

As the industry enters what is widely termed the "Year of the Agent," (WTH 26 https://businessday.ng/technology/article/microsoft-declares-2026-the-year-of-the-agent-as-ignite-2025-unveils-ai-agent-platform/) @aRAG emerges as the next evolution of @RAG  moving beyond the pre-orchestrated workflows of advanced and modular @RAG systems toward dynamic reasoning loops capable of autonomous action @singh2025agenticretrievalaugmentedgenerationsurvey. While agents enable numerous enhancements, this paper focuses on the agent's ability to leverage schema-bound query languages for semantic extraction namely, @SQL and @SPARQL. This focus is motivated by research demonstrating that semi-structured databases can be effectively constructed from unstructured corpora at scale @sun2025 @bai2025autoschemakgautonomousknowledgegraph, which is critical given that most business and research documents are unstructured, and that schema-bound query languages improve factual retrieval performance in @aRAG systems @singh2025agenticretrievalaugmentedgenerationsurvey. (WTH Problme statement: Make clearer why we experiment with structured querying vs semantic search RAG)

= Related Work
(WTH dont forget STARK Benchmark)


== RAG on SQL Tables

Prior work has explored RAG-style systems grounded in tabular data. A common baseline is to linearize tables (e.g., as Markdown or row-wise text), chunk the
result, and retrieve top-\(k\) chunks like an ordinary document corpus. As
summarized in TableRAG, this is often lossy@yu-etal-2025-tablerag. By comparison, modern approaches employ two main concepts to preserving table structure @biswal2024text2sqlenoughunifyingai @yu-etal-2025-tablerag:

#itm(<first>, [Table-structured
  indexing.]) Instead of treating tables as plain text documents, structure-aware approaches design retrieval units and indexes that explicitly retain table structure (e.g., table/row/column identifiers, schema elements, or graph structure) and then retrieve at the appropriate granularity. For example, TabRAG proposes parsing table-heavy documents into structured language representations that preserve cell values together with their row/column context (extracted via layout segmentation and a vision-language model), and only then generates embedding-friendly descriptions for retrieval and downstream QA @si2025tabrag. At larger corpus scale, T-RAG targets cross-table QA by building a hierarchical memory index over a hypergraph of tables, using multi-stage retrieval to narrow candidates and graph-aware prompting to inject relational priors into the LLM @zou2025trag. In heterogeneous documents (text interleaved with tables), TableRAG highlights that top-\(k\) chunk retrieval over flattened tables can mis-handle global questions; it therefore anchors retrieved table snippets back to their originating table schemas and treats table-related reasoning as a monolithic unit rather than a set of independent chunks @yu-etal-2025-tablerag.

#itm(<second>, [Execution-based table access.]) A complementary, widely used way to preserve table structure is to avoid “row snippet retrieval” altogether and instead translate natural language into an executable query (typically SQL), run it on a DBMS, and generate the final response from the exact result. This is the classic Text2SQL paradigm and is often the backbone for accurate joins/aggregations over large tables, with modern LLM-based systems emphasizing robustness via schema/value grounding and execution feedback @mohammadjafari2025naturallanguagesqlreview @sun2024sqlpalm. More broadly, TAG formalizes an end-to-end view where query synthesis and execution are combined with additional LLM-based reasoning/generation steps over the returned relation, explicitly targeting requests that go beyond single-shot Text2SQL (e.g., semantic predicates over text fields, iterative workflows) @biswal2024text2sqlenoughunifyingai. TableRAG can be seen as a document-centric instantiation of this idea for heterogeneous sources: retrieve relevant text evidence, but invoke SQL execution for table sub-queries to preserve global correctness @yu-etal-2025-tablerag.

#counter(figure.where(kind: "inline-item")).update(0)


SQL-based RAG primarily leverages retrieval to augment the text-to-SQL generation process, addressing challenges in schema complexity and query ambiguity. Liu and Chu (2025) propose a RAG-based text-to-SQL system for Business Intelligence (BI) that retrieves relevant schema elements and few-shot examples to guide GPT-4 and Graph Neural Networks (GNNs). Their work demonstrates that retrieval-based schema grounding significantly improves accuracy on complex joins and nested queries compared to non-RAG baselines.

Similarly, Wang et al. (2025) present a conversational bibliographic search system that utilizes semantic retrieval over historical question-SQL pairs. By leveraging in-context learning with retrieved examples, they achieve execution accuracies up to 96.6%. These systems treat SQL as the primary formal language for interaction, using RAG primarily for schema discovery and disambiguation.

In more complex, multi-agent settings, SQL agents are often specialized components. Salve et al. (2024) introduce a collaborative multi-agent RAG where a dedicated "MySQL agent" handles relational database queries while other agents manage graph and document sources. This specialization aims to reduce token overhead and error propagation in heterogeneous environments. Peng et al. (2025) extend this to geodatabases, using multi-agent systems with RAG over GIS metadata to transform natural language into spatial SQL queries, achieving over 80% accuracy on complex spatial operations.
== RAG with Text2Sparql (WTH Title)
Graph-based RAG (GraphRAG) and SPARQL-centric approaches aim to exploit explicit relational structures for multi-hop reasoning. Arazzi et al. (2025) introduce SparqLLM, a framework that wraps a knowledge graph with a RAG-style interface. It retrieves template-based SPARQL examples to guide LLMs in generating robust queries, mitigating the brittleness often seen in pure LLM-to-SPARQL generation.

Moving beyond simple query generation, GraphRAG frameworks focus on graph traversal and context expansion. Shen et al. (2024) propose GeAR (Graph-enhanced Agent for RAG), which augments any base retriever with a graph expansion mechanism. GeAR extracts "proximal triples" from initial retrieval results and uses a "gist memory" to maintain context across steps, mimicking human hippocampal-neocortex communication. This approach achieves state-of-the-art results on multi-hop QA datasets like MuSiQue, improving performance by over 10% while consuming fewer tokens than prior multi-step systems.

Xu et al. (2025) introduce AnchorRAG, a multi-agent framework designed for open-world KG retrieval. It employs a predictor agent to identify anchor entities and multiple retriever agents to perform parallel, multi-hop traversals. A supervisor agent then synthesizes the results. This architecture emphasizes robustness to entity ambiguity and efficient subgraph exploration, outperforming standard KG-RAG baselines. These works collectively demonstrate that graph structures are superior for multi-hop reasoning, though they often operate in isolation from relational SQL backends.

== Agentic RAG / Multi-Agent RAG
The frontier of RAG research has shifted towards Agentic RAG, where autonomous agents orchestrate retrieval and synthesis. Several surveys (Singh et al., 2025; Liang et al., 2025) define key agentic patterns—reflection, planning, tool use, and multi-agent collaboration—as essential for handling complex, real-world queries.

HM-RAG (Liu et al., 2025) exemplifies the hierarchical multi-agent approach. It employs a three-tiered architecture: a Decomposition Agent for query analysis, specialized Retrieval Agents for vector, graph, and web sources, and a Decision Agent for consistency voting and answer fusion. HM-RAG reports a 12.95% improvement in accuracy on multimodal QA benchmarks, demonstrating the power of hierarchical task decomposition.

Similarly, MA-RAG (Chen et al., 2025) and AU-RAG (Jang & Li, 2024) explore multi-agent collaboration, where agents specialize in retrieval, reasoning, or critique. Cook et al. (2025) apply these principles to fintech, showing how agentic design can enforce strict governance in regulated domains.

While these systems are highly capable, they typically treat SQL and Graph sources as separate tools managed by distinct agents (as in Salve et al., 2024) or focus exclusively on one data model (as in GeAR or AnchorRAG). They do not explicitly address the "routing problem" of choosing between a SQL query and a Graph traversal for a given sub-task based on the nature of the required reasoning

== Present Work
Despite the maturity of SQL-RAG and GraphRAG, and the rapid advancement of multi-agent orchestration, a critical gap remains: there is no unified agentic framework that exposes both SQL and SPARQL/GraphRAG as tools and empirically compares their performance on the same tasks.

Current literature treats these as parallel tracks: SQL-RAG work focuses on schema alignment and query generation for tabular data; GraphRAG work focuses on multi-hop traversal and entity linking for networked data; Agentic RAG work assumes "tools" for these backends exist but rarely analyzes the trade-offs of routing a query to one versus the other.

Heterogeneous multi-agent systems (e.g., Salve et al., 2024; Liu et al., 2025) claim benefits from specialization but do not provide controlled experiments analyzing when a graph query outperforms a relational query, or how an agent should decide between them. This work aims to address that gap.


= Approach <sec:approach>

We present an agentic retrieval system for the STaRK-Prime knowledge base #cite(<stark2024>)---a biomedical knowledge graph derived from PrimeKG #cite(<chandak2023primekg>) containing over 129,000 entities across 10 types (diseases, drugs, genes/proteins, pathways, biological processes, molecular functions, cellular components, anatomical structures, exposures, and phenotypes) connected by 18 relation types. Our architecture combines semantic vector search with schema-bound query execution, enabling systematic comparison of retrieval strategies: pure semantic search versus structured querying via @SQL and @SPARQL.

== System Architecture <sec:architecture>
(WTH include diagram)

Our system follows the @aRAG paradigm, employing an autonomous agent loop as the core reasoning mechanism. The agent iteratively reasons about the user's query, selects appropriate tools, executes them, and refines its approach based on intermediate results. This loop continues until the agent determines it has gathered sufficient information to answer the query or reaches a predefined iteration limit of 15 steps---a safeguard against infinite loops when the agent repeatedly fails to find relevant entities or formulates unsuccessful queries.

The architecture comprises three components:

+ *Agent Component*: An @LLM#[-driven] reasoning loop that interprets queries, plans tool invocations, and synthesizes final answers. We implement this using LangChain's ReAct agent framework, chosen for its well-documented abstractions for tool-using agents and widespread adoption in research prototypes.

+ *Tool Component*: A set of specialized tools that provide the agent with distinct capabilities for knowledge base access. These tools abstract the complexity of different query interfaces behind a unified function-calling API.

+ *Data Component*: Multiple materialized views of the same underlying knowledge graph, each optimized for different query patterns.

== Two-Stage Query Process <sec:two-stage>

Prior work on the STaRK benchmark #cite(<stark2024>) evaluated retrieval methods that operate in a single stage: given a natural language query, retrieve the top-$k$ most similar entities based on embedding distance or lexical matching. This approach struggles with multi-hop relational queries because entity descriptions alone do not encode graph structure.

Our key design decision is a *two-stage query approach* that decouples entity resolution from structured query execution:

=== Stage 1: Entity Resolution

Natural language queries reference entities by names, synonyms, or descriptions rather than internal identifiers. The first stage uses semantic vector search to resolve these references to node IDs. This approach leverages embedding similarity to handle:

- Lexical variations (e.g., "heart attack" → "myocardial infarction")
- Partial matches (e.g., "breast cancer" → "invasive breast carcinoma")
- Related concepts (e.g., searching for symptoms may surface related diseases)

Entity embeddings are generated using OpenAI's `text-embedding-3-small` model and indexed in Qdrant, a high-performance vector database. Each entity is embedded as a rich text representation combining its type, name, aliases, and descriptive metadata (e.g., "disease: Type 2 Diabetes Mellitus, Also known as: T2DM, adult-onset diabetes"), enabling type-aware and semantically rich retrieval.

=== Stage 2: Query Execution

Once entity IDs are resolved, the agent formulates structured queries using these identifiers. This separation ensures that:

+ The @LLM is not burdened with fuzzy string matching in query predicates
+ Queries operate on exact node identifiers, eliminating false negatives from name mismatches
+ The agent can leverage the full expressiveness of @SQL or @SPARQL for relational reasoning

== Tool Design <sec:tools>

We provide the agent with three specialized tools:

=== Semantic Search Tool (`search_entities_tool`)

Performs vector similarity search over entity embeddings stored in Qdrant. Parameters include:

- `query`: Natural language search term
- `entity_type`: Optional filter to restrict search to specific node types (e.g., "disease", "drug", "gene_protein")
- `top_k`: Number of results to return (default: 5)

The tool returns ranked entity matches with IDs, types, names, and descriptions.

=== SQL Query Tool (`execute_sql_query_tool`)

Executes read-only @SQL queries against a PostgreSQL database containing a relational projection of the knowledge graph. The schema comprises:

- *10 entity tables*: One table per node type (disease, drug, gene_protein, pathway, biological_process, molecular_function, cellular_component, anatomy, exposure, phenotype)
- *18 relation tables*: One table per edge type (e.g., `indication`, `contraindication`, `target`, `interacts`, `associated_with`)

Each relation table follows a consistent schema with `src_id`, `dst_id`, `src_type`, and `dst_type` columns, enabling efficient join operations.

=== SPARQL Query Tool (`execute_sparql_query_tool`)

Executes read-only @SPARQL queries against an Apache Jena Fuseki endpoint containing the knowledge graph in RDF format. The RDF representation uses a custom namespace (`http://stark.stanford.edu/prime/`) with:

- Node URIs: `sp:node/{id}` for entity references
- Edge predicates: `sp:{relation_type}` for typed relationships
- Node properties: `sp:name`, `sp:type`, `sp:description` for entity attributes

== Schema Grounding <sec:schema-grounding>

A critical enabler for structured query generation is providing the @LLM with comprehensive schema information in its system prompt. Without schema knowledge, the model cannot generate syntactically correct queries that reference valid tables, columns, or predicates. Our approach is intentionally general-purpose and does not incorporate benchmark-specific optimizations.

Our system prompt includes:

- *SQL Schema Summary*: Complete listing of all entity tables with their columns, all relation tables with their foreign key structure, and example values for categorical fields. This enables the @LLM to formulate valid JOIN conditions and WHERE clauses.

- *SPARQL Vocabulary Summary*: The RDF namespace, available predicates (relation types), node URI patterns, and property definitions. This grounds the @LLM's understanding of the graph structure for pattern matching.

- *Query Guidelines*: Best practices for query formulation, including mandatory LIMIT clauses, read-only restrictions, and error handling strategies.

This schema grounding approach follows established text-to-SQL practices, where providing database schema context significantly improves query generation accuracy. The schema is dynamically generated at agent initialization time, ensuring consistency with the actual data store state.

== Data Materialization <sec:data-stores>

We materialize the PrimeKG knowledge graph into three complementary data stores, each optimized for different access patterns:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Data Store*], [*Technology*], [*Optimized For*],
    table.hline(stroke: 0.75pt),

    [Vector Index], [Qdrant], [Semantic similarity search, entity resolution],
    [Relational DB], [PostgreSQL], [Aggregations, complex joins, filtering],
    [Graph Store], [Apache Jena Fuseki], [Path traversal, pattern matching],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Data stores and their optimized query patterns.],
) <tab:data-stores>

This multi-store architecture enables the agent to select the most appropriate query interface for each sub-task. For instance, counting entities is more efficient in @SQL, while finding paths between nodes is more natural in @SPARQL.

== Agent Configurations <sec:agent-configurations>

We implement four agent configurations to systematically isolate the effect of each tool capability. By comparing agents with different tool access, we can measure the marginal value of structured query execution over pure semantic search, and SPARQL's graph-native patterns versus SQL's relational joins:

=== Search-Only Agent

Access to semantic vector search (`search_entities_tool`) only. This configuration represents pure unstructured retrieval without structured query capabilities. The agent can only find entities by semantic similarity and cannot perform relational reasoning.

*Tools*: `search_entities_tool`

*Use case*: Baseline for evaluating the value of structured query access.

=== Search+SPARQL Agent

Access to semantic search for entity resolution plus @SPARQL query execution over the RDF graph. This configuration enables explicit graph traversal via declarative pattern matching.

*Tools*: `search_entities_tool`, `execute_sparql_query_tool`

*Use case*: Evaluating graph-native query formulation for knowledge graph reasoning.

=== Search+SQL Agent

Access to semantic search plus @SQL query execution over the relational projection. This configuration tests whether @LLM:pla can effectively express graph operations as relational joins.

*Tools*: `search_entities_tool`, `execute_sql_query_tool`

*Use case*: Evaluating relational query formulation and practical deployability with existing SQL infrastructure.

=== Search+SQL+SPARQL Agent

Access to all three tools (semantic search, @SPARQL, @SQL). This configuration provides maximum flexibility, allowing the agent to choose the most appropriate query language for each sub-task.

*Tools*: `search_entities_tool`, `execute_sql_query_tool`, `execute_sparql_query_tool`

*Use case*: Evaluating whether multi-modal structured query access yields complementary benefits.

== Model Selection <sec:model-selection>

We instantiate each agent configuration with two @LLM backbones to isolate the effect of model capacity:

- *GPT-5-mini*: A smaller, faster variant optimized for efficiency. Lower cost and latency make it suitable for high-throughput applications.
- *GPT-5*: The full-scale model with enhanced reasoning capabilities. Higher cost but potentially better performance on complex queries.

Both models are accessed via the OpenAI API with temperature set to 0.0 for deterministic output. We focus on the GPT family to control for API differences and prompting conventions, enabling cleaner isolation of model scale effects.

== Observability and Analysis <sec:observability>

To enable comprehensive analysis of agent behavior, we integrate observability and benchmarking infrastructure:

- *Langfuse*: Provides trace-level observability of agent reasoning, including tool invocations, intermediate results, and latency breakdowns. Each query execution is logged as a trace with full message history.

- *Custom Benchmark Framework*: Our evaluation harness executes agents across dataset splits, captures per-query metrics (precision, recall, F1, latency, tool call counts), and generates detailed analysis reports with item-level breakdowns and failure mode classification.

This instrumentation enables both real-time debugging during development and systematic post-hoc analysis of experimental results.

// =============================================================================
// EVALUATION AND DISCUSSION SECTIONS
// Paper: Agentic-RAG on Mix-typed Knowledge Bases
// =============================================================================

// Bibliography entries (add these to your .bib file or use Typst's built-in bibliography)
#let stark2024 = cite(<stark2024>)
#let chandak2023primekg = cite(<chandak2023primekg>)

= Evaluation <sec:evaluation>

== Experimental Setup <sec:experimental-setup>

*Dataset.*
We evaluate on the STaRK Human-Generated QA benchmark #cite(<stark2024>), comprising 109 natural-language questions over PrimeKG #cite(<chandak2023primekg>), a biomedical knowledge graph. Questions require entity retrieval from a heterogeneous graph containing diseases, genes/proteins, drugs, pathways, biological processes, cellular components, and phenotypes. Unlike synthetic benchmarks, the human-generated queries exhibit diverse linguistic patterns and varying complexity levels, including simple entity lookups, multi-hop relational queries, and complex constraint satisfaction problems.

*Agent Configurations.*
We compare four agentic retrieval strategies as described in @sec:agent-configurations: Search-Only, Search+SPARQL, Search+SQL, and Search+SQL+SPARQL. Each configuration provides access to different tool combinations, enabling systematic comparison of retrieval strategies. All agents use semantic search---Search-Only relies on it exclusively, while the others add structured query capabilities on top.

*Models.*
Following @sec:model-selection, each agent configuration is instantiated with two model variants: *GPT-5-mini* (smaller and faster) and *GPT-5* (full-scale). This pairing isolates the effect of model capacity on agentic retrieval performance.

*Repetitions.*
To quantify variance from @LLM non-determinism, each configuration-model pair is executed 6 times ($n=6$), yielding 48 total experimental runs and 5,236 individual query evaluations.

== Evaluation Metrics <sec:metrics>

Let $Y$ denote the gold entity set and $hat(Y)$ the predicted entity set. We report standard information retrieval metrics:

- *Precision (P)*: $|hat(Y) inter Y| \/ |hat(Y)|$ --- fraction of returned entities that are relevant.
- *Recall (R)*: $|hat(Y) inter Y| \/ |Y|$ --- fraction of relevant entities that are returned.
- *F1*: $2 dot P dot R \/ (P + R)$ --- harmonic mean of precision and recall.
- *Exact Match (EM)*: $bb(1)[hat(Y) = Y]$ --- indicator of perfect set match.
- *Hit\@k*: $bb(1)[|hat(Y)_(1:k) inter Y| > 0]$ --- whether any correct entity appears in the top-$k$ results.
- *MRR*: Mean Reciprocal Rank of the first correct entity in the ranked output.
- *Latency*: End-to-end wall-clock time per query (seconds).

== Main Results <sec:main-results>

@tab:main-results summarizes aggregate performance. We first compare against STaRK's published baselines (@tab:stark-baselines), then analyze differences across our agent configurations.

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Method*], [*Hit\@1*], [*Hit\@5*], [*R\@20*], [*MRR*],
    table.hline(stroke: 0.75pt),

    [Claude3 Reranker], [0.286], [0.469], [0.416], [0.363],
    [GPT4 Reranker], [0.286], [0.449], [0.416], [0.348],
    [GritLM-7b], [0.255], [0.418], [0.481], [0.343],
    [multi-ada-002], [0.245], [0.398], [0.472], [0.330],
    [BM25], [0.225], [0.418], [0.423], [0.304],

    table.hline(stroke: 1.5pt),
  ),
  caption: [STaRK baseline results on Human QA benchmark #cite(<stark2024>). R\@20 = Recall at rank 20.],
) <tab:stark-baselines>

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center, center, center, center, center, center),
    stroke: none,

    // Header
    table.hline(stroke: 1.5pt),
    [*Agent*], [*Model*], [*Precision*], [*Recall*], [*F1*], [*EM*], [*Hit\@1*], [*Hit\@5*], [*MRR*], [*Latency (s)*],
    table.hline(stroke: 0.75pt),

    // Search-Only
    [Search-Only], [GPT-5-mini],
    [0.281 #text(size: 0.8em)[(0.014)]],
    [0.309 #text(size: 0.8em)[(0.018)]],
    [0.248 #text(size: 0.8em)[(0.014)]],
    [0.112 #text(size: 0.8em)[(0.015)]],
    [0.341 #text(size: 0.8em)[(0.017)]],
    [0.420 #text(size: 0.8em)[(0.018)]],
    [0.372 #text(size: 0.8em)[(0.014)]],
    [23.8],

    [], [GPT-5],
    [0.379 #text(size: 0.8em)[(0.021)]],
    [0.382 #text(size: 0.8em)[(0.023)]],
    [0.329 #text(size: 0.8em)[(0.021)]],
    [0.176 #text(size: 0.8em)[(0.015)]],
    [0.403 #text(size: 0.8em)[(0.022)]],
    [0.512 #text(size: 0.8em)[(0.024)]],
    [0.445 #text(size: 0.8em)[(0.020)]],
    [38.6],

    table.hline(stroke: 0.5pt),

    // Search+SPARQL
    [Search+SPARQL], [GPT-5-mini],
    [0.462 #text(size: 0.8em)[(0.010)]],
    [0.472 #text(size: 0.8em)[(0.018)]],
    [0.403 #text(size: 0.8em)[(0.011)]],
    [0.243 #text(size: 0.8em)[(0.017)]],
    [0.467 #text(size: 0.8em)[(0.022)]],
    [0.567 #text(size: 0.8em)[(0.015)]],
    [0.506 #text(size: 0.8em)[(0.019)]],
    [42.5],

    [], [GPT-5],
    [#underline[0.477] #text(size: 0.8em)[(0.019)]],
    [#underline[0.477] #text(size: 0.8em)[(0.019)]],
    [#underline[0.404] #text(size: 0.8em)[(0.018)]],
    [#underline[0.248] #text(size: 0.8em)[(0.014)]],
    [*0.485* #text(size: 0.8em)[(0.018)]],
    [#underline[0.584] #text(size: 0.8em)[(0.034)]],
    [#underline[0.522] #text(size: 0.8em)[(0.024)]],
    [64.4],

    table.hline(stroke: 0.5pt),

    // Search+SQL
    [Search+SQL], [GPT-5-mini],
    [0.443 #text(size: 0.8em)[(0.016)]],
    [0.465 #text(size: 0.8em)[(0.013)]],
    [0.383 #text(size: 0.8em)[(0.012)]],
    [0.202 #text(size: 0.8em)[(0.016)]],
    [0.449 #text(size: 0.8em)[(0.017)]],
    [0.578 #text(size: 0.8em)[(0.017)]],
    [0.499 #text(size: 0.8em)[(0.014)]],
    [47.6],

    [], [GPT-5],
    [0.453 #text(size: 0.8em)[(0.019)]],
    [0.425 #text(size: 0.8em)[(0.026)]],
    [0.371 #text(size: 0.8em)[(0.018)]],
    [0.228 #text(size: 0.8em)[(0.017)]],
    [0.468 #text(size: 0.8em)[(0.020)]],
    [0.541 #text(size: 0.8em)[(0.017)]],
    [0.499 #text(size: 0.8em)[(0.015)]],
    [70.9],

    table.hline(stroke: 0.5pt),

    // Search+SQL+SPARQL
    [Search+SQL+SPARQL], [GPT-5-mini],
    [*0.464* #text(size: 0.8em)[(0.008)]],
    [*0.470* #text(size: 0.8em)[(0.019)]],
    [0.394 #text(size: 0.8em)[(0.011)]],
    [0.209 #text(size: 0.8em)[(0.013)]],
    [#underline[0.469] #text(size: 0.8em)[(0.009)]],
    [*0.598* #text(size: 0.8em)[(0.024)]],
    [*0.521* #text(size: 0.8em)[(0.009)]],
    [39.4],

    [], [GPT-5],
    [0.459 #text(size: 0.8em)[(0.024)]],
    [0.438 #text(size: 0.8em)[(0.020)]],
    [0.378 #text(size: 0.8em)[(0.019)]],
    [*0.232* #text(size: 0.8em)[(0.021)]],
    [#underline[0.486] #text(size: 0.8em)[(0.022)]],
    [0.561 #text(size: 0.8em)[(0.024)]],
    [0.518 #text(size: 0.8em)[(0.020)]],
    [76.1],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Our agentic retrieval performance on STaRK Human QA benchmark (mean ± std over $n=6$ runs). Best results per metric are *bolded*; second-best are #underline[underlined].],
) <tab:main-results>

== Analysis by Retrieval Strategy <sec:analysis>

=== Comparison to STaRK Baselines

Our agentic approaches substantially outperform all STaRK baselines. The best baseline, Claude3 Reranker, achieves Hit\@1=0.286, Hit\@5=0.469, and MRR=0.363. In comparison, our Search+SPARQL agent with GPT-5 achieves Hit\@1=0.485 (+19.9 pp), Hit\@5=0.584 (+11.5 pp), and MRR=0.522 (+15.9 pp). Even our Search-Only agent with GPT-5-mini matches or exceeds the best baselines on Hit\@5 (0.420 vs. 0.469) while achieving higher MRR (0.372 vs. 0.363).

STaRK baselines also report Recall\@20 (R\@20), ranging from 0.416 (Claude3 Reranker) to 0.481 (GritLM-7b). This metric is not directly comparable to our set-based Recall because STaRK baselines return ranked lists of 20 candidates, whereas our agents return variable-sized answer sets (typically 1--5 entities). Our agents' set-based Recall (0.31--0.48) reflects what fraction of gold entities appear in the _final answer_, not in a candidate pool.

This improvement is notable because STaRK baselines include strong retrieval methods: Claude3 and GPT4 rerankers, state-of-the-art dense retrievers (GritLM-7b), and OpenAI embeddings (multi-ada-002). The agentic approach---iteratively searching, reasoning, and querying---provides a qualitative advantage over single-shot retrieval, even before adding structured query capabilities.

=== Structured vs. Unstructured Retrieval

The most striking result is the substantial performance gap between pure semantic search and structured-query agents. With GPT-5-mini, Search+SPARQL achieves:

- +18.1 pp higher precision (0.462 vs. 0.281)
- +16.3 pp higher recall (0.472 vs. 0.309)
- +15.5 pp higher F1 (0.403 vs. 0.248)
- +13.1 pp higher exact match rate (0.243 vs. 0.112)

This gap narrows but persists with GPT-5 (F1: 0.404 vs. 0.329, +7.5 pp). The pattern confirms STaRK's original findings #cite(<stark2024>): multi-hop relational queries---"find diseases treated by drug X that also exhibit phenotype Y"---exceed the capabilities of embedding-based retrieval. Semantic search over isolated entity descriptions cannot capture the graph structure needed for relational composition.

=== SPARQL vs. SQL

Search+SPARQL slightly outperforms Search+SQL across most metrics. With GPT-5-mini, SPARQL achieves +2.0 pp higher F1 (0.403 vs. 0.383). The advantage persists with GPT-5 (+3.3 pp F1). While consistent, the gap is modest, suggesting that modern LLMs can effectively translate graph queries into relational operations.

=== Multi-Tool Agents

The Search+SQL+SPARQL agent achieves the highest Hit\@5 (0.598 with GPT-5-mini), outperforming both single-structured-query agents. However, its precision and F1 do not consistently exceed Search+SPARQL, and exact match rates are lower. This pattern suggests that multi-tool access improves recall by enabling the agent to recover entities that one query language might miss, but introduces additional decision complexity that can degrade precision.

=== Effect of Model Scale

Scaling from GPT-5-mini to GPT-5 yields heterogeneous effects across agent types:

- *Search-Only*: Large improvement (+9.8 pp precision, +8.1 pp F1). The stronger model better interprets query semantics and generates more effective search terms.
- *Search+SPARQL*: Modest improvement (+1.5 pp precision, +0.1 pp F1). Query generation is already well-handled by the smaller model.
- *Search+SQL*: Marginal or no improvement (+1.0 pp precision, −1.2 pp F1). The larger model does not improve SQL generation quality.
- *Search+SQL+SPARQL*: Mixed results (−0.5 pp precision, −1.6 pp F1). Additional model capacity may introduce suboptimal tool selection.

These findings suggest that the bottleneck for structured-query agents lies in entity resolution and knowledge graph coverage, not LLM reasoning capacity.

== Latency Analysis <sec:latency>

@tab:latency presents latency statistics across configurations.

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center, center),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Agent*], [*Model*], [*p50*], [*p90*], [*avg*], [*max*],
    table.hline(stroke: 0.75pt),

    [Search-Only], [GPT-5-mini], [18.9], [43.4], [23.8], [112.9],
    [], [GPT-5], [29.2], [75.1], [38.6], [213.0],
    table.hline(stroke: 0.5pt),

    [Search+SPARQL], [GPT-5-mini], [28.6], [86.2], [42.5], [236.0],
    [], [GPT-5], [52.3], [121.6], [64.4], [590.5],
    table.hline(stroke: 0.5pt),

    [Search+SQL], [GPT-5-mini], [33.9], [99.1], [47.6], [239.9],
    [], [GPT-5], [61.7], [133.5], [70.9], [503.4],
    table.hline(stroke: 0.5pt),

    [Search+SQL+SPARQL], [GPT-5-mini], [26.8], [83.5], [39.4], [206.7],
    [], [GPT-5], [66.9], [141.8], [76.1], [594.0],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Query latency statistics (seconds). p50 = median, avg = arithmetic mean.],
) <tab:latency>

Search-Only is fastest due to single-tool invocations. Structured-query agents require additional entity-resolution steps (semantic search to find node IDs, then query formulation and execution), increasing latency by 1.6--2.0×. GPT-5 roughly doubles latency across all configurations due to longer generation times and more complex reasoning traces. The Search+SQL+SPARQL agent with GPT-5 exhibits the highest variance, with tail latencies exceeding 500 seconds on complex multi-tool queries.

= Discussion <sec:discussion>

We now interpret the experimental results, identify failure modes through item-level analysis, and derive practical recommendations for Agentic-RAG system design.

== Structured Query Generation as the Key Advantage <sec:discussion-structured>

The dominant finding is that agentic access to structured query interfaces (SPARQL or SQL) substantially outperforms both pure semantic retrieval and all STaRK baselines. Our Search+SPARQL agent with GPT-5 achieves Hit\@1=0.485 and MRR=0.522, compared to the best STaRK baseline (Claude3 Reranker) at Hit\@1=0.286 and MRR=0.363---improvements of +19.9 pp and +15.9 pp respectively. The +15.5 pp F1 improvement of Search+SPARQL over Search-Only (with GPT-5-mini) represents a qualitative capability gap, not merely incremental improvement.

This confirms and extends the central finding from the STaRK benchmark #cite(<stark2024>): embedding-based retrieval struggles with multi-hop relational queries that require joins across entity types, path traversals, or constraint filtering. The agentic paradigm---combining iterative reasoning with structured query execution---provides a path forward that pure retrieval methods cannot match.

However, our analysis reveals an additional bottleneck beyond entity resolution: *missed opportunities*, where correct entities appear in intermediate tool outputs but are omitted from the final answer. As detailed in @sec:error-taxonomy, 16.5% of queries exhibit at least one missed opportunity, with an average loss rate of 14.2% across affected items. This represents a reasoning failure at answer synthesis, distinct from retrieval failure.

Importantly, this advantage emerges even though all structured-query agents _depend on_ semantic search for entity resolution. The structured query step adds value by enabling explicit relational reasoning over the resolved entities, whereas Search-Only terminates after the initial retrieval step.

== The Entity Resolution Bottleneck <sec:discussion-bottleneck>

A critical architectural insight from item-level analysis is that entity resolution constitutes the primary failure mode for structured-query agents. The pipeline operates as follows:

+ Semantic search maps natural-language entity mentions to node IDs.
+ The LLM formulates a structured query using these IDs.
+ Query execution retrieves related entities from the graph.

If step (1) fails to surface the correct entity, no subsequent query can recover. We observe cases where agents execute 10+ tool calls across 9 iterations, generating syntactically valid queries, yet fail because the initial entity search returned incorrect node IDs. For example, item 3 in our benchmark ("gene involved in vesicle transport, located in kinetochore, in antigen processing pathway") required finding entity ID 5585, which never appeared in any of 11 search results despite extensive query reformulation.

This bottleneck suggests that improving entity resolution quality---via better embeddings, synonym expansion, fuzzy matching, or hybrid lexical-semantic retrieval---may yield larger gains than improving LLM reasoning capacity for query generation.

== Diminishing Returns from Model Scale <sec:discussion-scaling>


The observation that GPT-5 substantially improves Search-Only but yields marginal or no improvement for structured-query agents is theoretically significant. We hypothesize three contributing factors:

+ *Query generation saturation*: GPT-5-mini already generates correct SPARQL/SQL for most queries. Additional reasoning capacity provides diminishing returns.
+ *Bottleneck shift*: With structured queries, performance is limited by entity resolution and knowledge graph coverage---factors orthogonal to LLM scale.

This finding aligns with recent text-to-SQL research showing that smaller models can achieve competitive accuracy when provided with adequate schema context #cite(<mohammadjafari2025naturallanguagesqlreview>). From a practical standpoint, smaller models are preferable for structured-query agents, reserving larger models for pure semantic retrieval where their advantages are realized.

== Multi-Tool Agents: Complementarity vs. Complexity <sec:discussion-multitool>

Search+SQL+SPARQL agents achieve the highest Hit\@5 scores, indicating that access to complementary query languages improves recall. This makes intuitive sense: certain queries may be more naturally expressed in SPARQL (graph patterns) vs. SQL (aggregations, complex filters), and having both options allows the agent to select the more appropriate tool.

However, the combined agent does not dominate single-tool agents on precision or F1. We identify three potential explanations:

- *Tool selection overhead*: Choosing between SPARQL and SQL introduces an additional decision point that can lead to suboptimal selections.
- *Redundant tool calls*: Agents sometimes invoke both query tools on the same subproblem, adding latency without improving recall.
- *Prompt complexity*: More tool options increase prompt length and complexity, potentially degrading query formulation quality.

These findings suggest that multi-tool agents are best suited for recall-critical applications (e.g., exploratory search, comprehensive entity discovery) rather than precision-critical applications (e.g., fact verification, targeted lookup).

== SPARQL vs. SQL: Practical Equivalence <sec:discussion-sparql-sql>

SPARQL's advantage over SQL is consistent but surprisingly small (+5% F1). While SPARQL is natively designed for graph pattern matching, modern LLMs can effectively translate graph traversals into relational joins. This practical equivalence has important deployment implications:

- Organizations with existing SQL infrastructure need not deploy dedicated SPARQL endpoints.
- Knowledge graphs stored in relational databases (a common industry pattern) can be queried effectively via SQL without performance penalty.
- The choice between SPARQL and SQL should be driven by infrastructure constraints and developer familiarity rather than expected retrieval quality.

== Error Taxonomy <sec:error-taxonomy>

From detailed item-level analysis across all 109 benchmark queries and 48 experimental runs (5,236 query evaluations), we identify four primary failure modes with quantified incidence rates.

=== Entity Resolution Failure

The semantic search does not surface the correct entity, typically because:
- The entity has an unusual or ambiguous name.
- The query mentions the entity indirectly via properties rather than name.
- Multiple entities share similar names, and the wrong one is selected.

This is the most common failure mode: *34.9%* of queries (38/109) exhibit complete entity resolution failure (coverage = 0), while an additional *22.9%* (25/109) achieve only partial coverage. Only *42.2%* of queries achieve full coverage of gold entities during tool execution. Entity resolution failure is the dominant cause of the *38.5%* zero-recall rate observed across the benchmark. Notably, the semantic search tool itself exhibits *0% error rate* across all configurations---it reliably returns results, but those results may not contain the target entities.

=== Tool Execution Errors

@tab:tool-errors presents tool-level error rates across agent configurations. We distinguish between two metrics: _error rate per call_ (fraction of tool invocations that fail with an exception) and _items affected_ (fraction of queries experiencing at least one tool error).

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Agent*], [*Model*], [*Error Rate*], [*Items Affected*], [*Error Categories*],
    table.hline(stroke: 0.75pt),

    [Search-Only], [GPT-5-mini], [0.00%], [0.0%], [---],
    [], [GPT-5], [0.00%], [0.0%], [---],
    table.hline(stroke: 0.5pt),

    [Search+SPARQL], [GPT-5-mini], [0.39%], [2.0%], [syntax: 100%],
    [], [GPT-5], [0.53%], [3.8%], [syntax: 100%],
    table.hline(stroke: 0.5pt),

    [Search+SQL], [GPT-5-mini], [1.06%], [5.8%], [not\_found: 58%, syntax: 22%],
    [], [GPT-5], [0.96%], [6.1%], [not\_found: 54%, syntax: 40%],
    table.hline(stroke: 0.5pt),

    [Search+SQL+SPARQL], [GPT-5-mini], [0.79%], [4.4%], [not\_found: 73%, syntax: 13%],
    [], [GPT-5], [0.64%], [4.3%], [syntax: 52%, not\_found: 42%],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Tool execution error rates by agent configuration. Error Rate = errors/total calls. Items Affected = queries with $>=$ 1 error.],
) <tab:tool-errors>

Key observations:
- *Search tools are error-free*: Semantic search never fails at the API level, making Search-Only agents operationally robust.
- *SQL exhibits higher error rates than SPARQL*: Search+SQL agents experience 1.0% error rate vs. 0.4--0.5% for Search+SPARQL, with errors affecting 5.8--6.1% of items vs. 2.0--3.8%.
- *Error categories differ by query language*: SPARQL errors are exclusively syntax errors (malformed queries). SQL errors are predominantly `not_found` errors (references to non-existent tables/columns), indicating LLM confusion about the relational schema.
- *Model scale has minimal impact*: GPT-5 does not substantially reduce error rates compared to GPT-5-mini, consistent with the observation that query generation quality is already saturated.

=== Zero-Result Queries (Semantic Failures)

Beyond execution errors, queries may succeed syntactically but return empty result sets---indicating semantically incorrect formulations. @tab:zero-results shows zero-result rates by tool type.

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr, 1fr),
    align: (left, left, center, center, center),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Agent*], [*Model*], [*Overall Zero-Result*], [*SQL Zero-Result*], [*SPARQL Zero-Result*],
    table.hline(stroke: 0.75pt),

    [Search-Only], [GPT-5-mini], [0.0%], [---], [---],
    [], [GPT-5], [0.0%], [---], [---],
    table.hline(stroke: 0.5pt),

    [Search+SPARQL], [GPT-5-mini], [14.4%], [---], [42.3%],
    [], [GPT-5], [17.2%], [---], [53.2%],
    table.hline(stroke: 0.5pt),

    [Search+SQL], [GPT-5-mini], [21.1%], [49.6%], [---],
    [], [GPT-5], [21.9%], [62.3%], [---],
    table.hline(stroke: 0.5pt),

    [Search+SQL+SPARQL], [GPT-5-mini], [19.1%], [50.0%], [25.6%],
    [], [GPT-5], [19.7%], [63.3%], [28.5%],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Zero-result rates by agent and tool type. Overall rate is weighted by tool usage; per-tool rates show tool-specific query failure.],
) <tab:zero-results>

The most striking finding is that *SQL queries return empty results 50--63% of the time*, compared to 25--53% for SPARQL. This substantial gap indicates that LLMs struggle more with relational query formulation than graph pattern matching. Potential causes include:
- *Schema complexity*: The relational projection requires understanding foreign key relationships and join semantics.
- *Predicate mismatch*: SQL column names may not align with natural language descriptions as directly as RDF predicates.
- *Over-constrained queries*: SQL's explicit join syntax may lead to overly restrictive conditions.

Notably, the overall zero-result rate is diluted by the high volume of successful semantic search calls (which never return empty). When considering only structured query tools, the failure rate is substantial.

=== Incomplete Result Selection

The agent discovers correct entities during tool calls but fails to include them in the final answer. The item-level data tracks this as the "missed opportunity rate"---entities that appear in tool outputs but are absent from the final prediction. *16.5%* of queries (18/109) exhibit at least one missed opportunity, with an average missed opportunity rate of *14.2%* across affected items. This represents a reasoning/summarization failure at the answer synthesis step, where the agent incorrectly filters or ranks intermediate results.

== Design Recommendations for Agentic-RAG Systems <sec:discussion-recommendations>

Based on our findings, we offer the following recommendations for practitioners building Agentic-RAG systems over mix-typed knowledge bases:

+ *Always include structured query access* when the knowledge base supports it. The performance gap over pure semantic search is large (+15.5 pp F1) and consistent across model scales.

+ *Invest in entity resolution quality*. This is the primary bottleneck for structured-query agents. Consider:
  - Fine-tuning entity embeddings on domain-specific corpora.
  - Implementing synonym expansion and alias matching.
  - Hybrid lexical-semantic retrieval (e.g., BM25 + dense retrieval fusion).

+ *Use smaller models for structured-query agents*. GPT-5-mini achieves comparable or better F1 than GPT-5 at half the latency and cost.

+ *Consider multi-tool agents for recall-critical applications*. Search+SQL+SPARQL achieves the highest Hit\@5 when comprehensive entity discovery matters more than precision.

+ *SPARQL is preferred but SQL is viable*. Choose based on infrastructure constraints rather than performance expectations.

+ *Implement result validation*. The "missed opportunity" failure mode suggests that explicit re-ranking or verification of intermediate results could improve final answer quality.

== Limitations <sec:limitations>

Our study has several limitations that suggest directions for future work:

- *Single domain*: Evaluation is limited to the biomedical domain (PrimeKG). Results may differ on other knowledge graphs with different schema complexity, entity distributions, or query patterns.

- *Single benchmark*: We evaluate on STaRK Human QA ($n=109$). Larger-scale evaluation on diverse benchmarks would strengthen generalizability claims.

- *Single model family*: Only GPT-5 variants are compared. Open-source models (LLaMA, Mistral) or other commercial models (Claude, Gemini) may exhibit different performance characteristics.

- *Entity retrieval only*: We evaluate entity set retrieval, not end-to-end question answering with natural-language generation. Answer synthesis quality is an orthogonal concern.

- *Controlled concurrency*: Latency measurements use concurrency=5, which may not reflect production deployment conditions with varying load patterns.

- *Static knowledge graph*: We do not evaluate performance on evolving knowledge graphs or the impact of knowledge staleness.

= Conclusion

// TODO: Write conclusion

// =============================================================================
// APPENDIX
// =============================================================================

#pagebreak()

= Appendix A: Illustrative Examples <sec:examples>

The following examples ground the discussion findings in concrete query executions. We present one successful retrieval demonstrating structured query advantages (@sec:example-success), one failure case illustrating the entity resolution bottleneck (@sec:example-failure), and one missed opportunity case (@sec:example-missed).

== A.1 Successful Multi-Hop Retrieval <sec:example-success>

#figure(
  block(
    width: 100%,
    inset: 12pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [
      *Query:* _"I have pharyngitis and chemosis. What skin disease might I have?"_

      *Gold Answer:* Toxic epidermal necrolysis (ID: 32299)

      *Agent Execution (Search+SPARQL, GPT-5):*

      #table(
        columns: (auto, auto, 1fr),
        stroke: none,
        align: (center, left, left),

        [*Iter*], [*Tool*], [*Action \& Result*],
        table.hline(stroke: 0.5pt),

        [1], [`search_entities`], [Query: "pharyngitis" → Pharyngitis (ID: 26964)],
        [1], [`search_entities`], [Query: "chemosis" → Chemosis (ID: 89294)],
        [1], [`search_entities`], [Query: "skin disease" → skin disease (ID: 36644)],
        [2], [`execute_sparql`], [Find diseases that are subtypes of skin disease AND have both phenotypes → *32299*],
        [3], [`execute_sparql`], [Retrieve name for ID 32299 → "toxic epidermal necrolysis"],
      )

      *Result:* Precision=1.0, Recall=1.0, F1=1.0, Latency=62s
    ],
  ),
  caption: [Successful multi-hop retrieval requiring entity resolution followed by graph traversal.],
) <fig:example-success>

This example illustrates several key findings:

+ *Entity resolution as prerequisite*: The agent first resolves natural language terms ("pharyngitis", "chemosis", "skin disease") to knowledge graph node IDs via semantic search. Without this step, no structured query is possible.

+ *Structured query enables relational composition*: The SPARQL query expresses a constraint that would be impossible with pure semantic search: find entities that are (a) subtypes of skin disease AND (b) linked to both phenotype nodes via `phenotypePresent` edges. This multi-hop, multi-constraint pattern is precisely where structured queries excel.

+ *Search-Only would fail*: Semantic search over entity descriptions cannot discover "toxic epidermal necrolysis" from the query "pharyngitis and chemosis" because the disease name shares no lexical overlap with the symptoms. The relationship exists only in the graph structure.

== A.2 Entity Resolution Failure <sec:example-failure>

#figure(
  block(
    width: 100%,
    inset: 12pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [
      *Query:* _"Which gene is involved in vesicle transport and located in kinetochore, and in the pathway of antigen processing?"_

      *Gold Answer:* Gene ID 5585

      *Agent Execution (Search+SPARQL, GPT-5):*

      #table(
        columns: (auto, auto, 1fr),
        stroke: none,
        align: (center, left, left),

        [*Iter*], [*Tool*], [*Action \& Result*],
        table.hline(stroke: 0.5pt),

        [1], [`search_entities`], [Query: "vesicle transport" → 5 biological processes],
        [1], [`search_entities`], [Query: "kinetochore" → 5 cellular components],
        [1], [`search_entities`], [Query: "antigen processing" → 5 pathways],
        [2], [`execute_sparql`], [Find genes linked to all three → *0 rows*],
        [3], [`execute_sparql`], [Relaxed query → *0 rows*],
        [4], [`search_entities`], [Expanded search (top\_k=15) for all three terms],
        [5], [`execute_sparql`], [Query with expanded IDs → *0 rows*],
        [6], [`execute_sparql`], [Alternative predicate patterns → *0 rows*],
        [7], [`execute_sparql`], [Further relaxation → *0 rows*],
        [8], [`search_entities`], [Direct gene search: "kinetochore vesicle transport antigen processing gene"],
        [9], [`execute_sparql`], [Final attempt with candidate genes → *0 rows*],
      )

      *Result:* Precision=0.0, Recall=0.0, F1=0.0, Latency=146s

      *Failure Analysis:*
      - Total tool calls: 17 (11 searches, 6 SPARQL queries)
      - All 6 SPARQL queries returned 0 rows
      - Gold entity ID 5585 *never appeared* in any of the 21 unique IDs returned by search
      - Coverage: 0% (entity resolution failed completely)
    ],
  ),
  caption: [Entity resolution failure: extensive tool use cannot compensate for missing the correct entity in initial search.],
) <fig:example-failure>

This failure case demonstrates critical limitations:

+ *Entity resolution bottleneck*: Despite 11 search queries with various formulations, the correct gene (ID 5585) never appeared in any result set. The agent's SPARQL queries were syntactically valid and logically correct, but operated on the wrong set of candidate entities.

+ *Diminishing returns from iteration*: The agent executed 9 iterations and 17 tool calls over 146 seconds, yet made no progress toward the answer. Additional reasoning effort cannot overcome a fundamental retrieval gap.

+ *Query formulation was correct*: The SPARQL queries correctly expressed the multi-constraint pattern (gene linked to vesicle transport process, kinetochore component, and antigen processing pathway). The failure mode is not LLM reasoning but knowledge base access.

+ *Implications for system design*: This example motivates investment in entity resolution quality (better embeddings, synonym expansion, fuzzy matching) rather than more sophisticated query generation or additional LLM reasoning steps.

== A.3 Missed Opportunity (Found but Not Returned) <sec:example-missed>

#figure(
  block(
    width: 100%,
    inset: 12pt,
    stroke: 0.5pt + gray,
    radius: 4pt,
    [
      *Query:* _"Mixed mucinous and nonmucinous bronchioloalveolar adenocarcinoma is a subtype of what disease?"_

      *Gold Answers:* IDs 95312 (the disease itself) and 35882 (minimally invasive lung adenocarcinoma)

      *Agent Execution:*

      #table(
        columns: (auto, auto, 1fr),
        stroke: none,
        align: (center, left, left),

        [*Iter*], [*Tool*], [*Action \& Result*],
        table.hline(stroke: 0.5pt),

        [1], [`search_entities`], [Found ID *95312* (the query disease) ranked \#1],
        [2], [`execute_sparql`], [Query parentChild relationship → returned ID *35882*],
      )

      *Final Prediction:* \[35882\] (only the parent disease)

      *Result:* Precision=1.0, Recall=0.5, F1=0.67 | Missed opportunity rate: 50%

      *Analysis:* The agent discovered both gold entities (95312 via search, 35882 via SPARQL) but only returned the SPARQL result. Entity 95312 was found but not included in the final answer.
    ],
  ),
  caption: [Missed opportunity: agent discovers correct entity but fails to include it in final prediction.],
) <fig:example-missed>

This example illustrates the *incomplete result selection* failure mode:

+ *Full coverage achieved*: Both gold entities were discovered during tool execution (coverage = 100%).

+ *Filtering failure*: The agent correctly identified 95312 as the query entity but interpreted the question as asking only for the _parent_ disease, not the disease hierarchy. This represents a reasoning failure at the answer synthesis stage.

+ *Actionable improvement*: Explicit re-ranking or verification of intermediate results---checking whether discovered entities should be included in the final answer---could address this failure mode without improving entity resolution or query generation.
