#import "@preview/charged-ieee:0.1.4": ieee

#set page(numbering: "1", number-align: center)

#show: ieee.with(
  title: [Agentic-RAG on Mix-typed Knowledge Base: A Comparative Study of SQL and SPARQL Querying],
  abstract: [
    Retrieval Augmented Generation (RAG) over structured knowledge bases remains limited when relying solely on semantic similarity search, which cannot capture relational structure. We present an agentic RAG system that combines semantic vector search with schema-bound query execution via SQL and SPARQL over the STaRK-Prime biomedical knowledge graph. Using a two-stage approach that decouples entity resolution from structured query execution, we systematically compare four agent configurations (Search-Only, Search+SQL, Search+SPARQL, Search+SQL+SPARQL) across two model scales (GPT-5-mini, GPT-5) on 109 human-generated questions. Structured query access yields a +15.5 percentage-point F1 improvement over pure semantic search, and all agentic configurations substantially outperform prior STaRK baselines. SPARQL slightly outperforms SQL (+2--3 pp F1), but the two languages are largely redundant ($r > 0.9$ item-level correlation). Scaling model capacity benefits Search-Only agents but yields diminishing returns for structured-query agents, whose primary bottleneck is entity resolution rather than query generation. We provide a detailed error taxonomy and design recommendations for practitioners building agentic RAG systems over semi-structured databases.
  ],
  authors: (
    (
      name: "Leonard Heininger",
      department: [ARDI],
      organization: [TU Berlin],
      location: [Berlin, Germany],
      email: "heininger@tu-berlin.de",
    ),
    (
      name: "Nicolas Kohl",
      department: [ARDI],
      organization: [TU Berlin],
      location: [Berlin, Germany],
      email: "nico-kohl@tu-berlin.de",
    ),
  ),
  index-terms: ("Agentic RAG", "Knowledge Graphs", "Text-to-SQL", "Text-to-SPARQL", "Biomedical QA"),
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
  ("DBMS", "Database Management System"),
)



= Introduction
@LLM:pla have fundamentally changed individual search behavior and information-seeking patterns. Whereas users previously needed to translate natural language into keyword searches tailored to algorithmic patterns, they can now directly employ @NLQ:pla to acquire information by interacting with @LLM systems. This approach better aligns with intuitive user behavior, enables more efficient information synthesis, and is already evident in observed user practices @sommerfeld2025ai @kaiser2025.

While @LLM:pla can answer @NLQ:pla directly from their parametric knowledge, this knowledge is static, bounded by the training cutoff, and prone to hallucination, particularly for domain-specific, rapidly evolving, or long-tail factual queries @gao2024retrievalaugmentedgenerationlargelanguage. @RAG addresses these limitations by integrating classical @IR into the generative workflow: retrieved documents ground the @LLM's output in verified sources, reducing hallucination and extending coverage beyond the model's parameters. Historically, @IR produced ranked document lists from unstructured corpora; @RAG repurposes this capability by feeding the top-ranked passages directly into the @LLM context, enabling answer synthesis rather than document ranking @gao2024retrievalaugmentedgenerationlargelanguage.

However, standard @RAG pipelines rely on semantic similarity search over text chunks, which is inherently limited when the underlying data is structured. Chunking tabular or graph-structured data into flat text fragments destroys explicit row/column dependencies, typed relationships, and multi-hop paths; as a result, semantic retrieval struggles with aggregation queries, constraint satisfaction, and compositional relational reasoning @yu-etal-2025-tablerag @biswal2024text2sqlenoughunifyingai. Schema-bound query languages such as @SQL and @SPARQL avoid these failure modes by operating directly on the relational or graph structure, leveraging exact joins, filters, and traversals that cannot be replicated by embedding-based retrieval alone.

As the industry enters what is widely termed the "Year of the Agent"#footnote[See, e.g., #link("https://medium.com/@billxu_atoms/why-2025-is-the-year-of-the-ai-agent-b2d5cef99f42")[B.\ Xu, _Why 2025 is the Year of the AI Agent_, Medium, 2025].], @aRAG emerges as the next evolution of @RAG, moving beyond pre-orchestrated workflows toward dynamic reasoning loops capable of autonomous action @singh2025agenticretrievalaugmentedgenerationsurvey. This paper investigates the agent's ability to leverage schema-bound query languages, specifically @SQL and @SPARQL, as retrieval tools alongside semantic search, and empirically compares these strategies on the same benchmark. The focus is motivated by two converging observations: first, that semi-structured databases can be effectively constructed from unstructured corpora at scale @sun2025 @bai2025autoschemakgautonomousknowledgegraph, making structured retrieval broadly applicable; and second, that schema-bound querying consistently improves factual retrieval performance over semantic search alone in agentic systems @singh2025agenticretrievalaugmentedgenerationsurvey.

= Related Work

== STaRK Benchmark

Our evaluation builds on the STaRK benchmark @stark2024, which pairs real-world knowledge graphs, including PrimeKG @chandak2023primekg, with natural-language queries whose gold answers are entity sets, enabling set-valued evaluation with standard @IR metrics (Hit\@$k$, MRR, Recall\@$k$). This protocol naturally frames entity retrieval as a @RAG task, since a system must identify _all_ relevant entities from a large candidate pool. Wu et al. show that embedding models and @LLM re-rankers degrade on queries requiring multi-hop relational reasoning, establishing a clear ceiling for unstructured retrieval @stark2024. AvaTaR @wu2024avatar exploits this gap by casting STaRK in an agentic setting: a tool-using @LLM agent optimized via contrastive reasoning achieves a 14\% relative improvement on Hit\@1 over non-agentic baselines. Our work extends this direction by adding schema-bound query execution (@SQL and @SPARQL) alongside semantic search and comparing these strategies head-to-head on STaRK-Prime.


== RAG on SQL Tables

Prior work has explored @RAG\-style systems grounded in tabular data. A common baseline is to parse tabular data into a text format, chunk the result, and retrieve top-\(k\) chunks like an ordinary document corpus. This is often lossy: it weakens explicit row/column dependencies, and chunking fragments surrounding context that is needed for aggregation-style or multi-hop queries @yu-etal-2025-tablerag @biswal2024text2sqlenoughunifyingai.

To avoid these failure modes, many recent approaches shift the retrieval step from semantic similarity search over serialized rows to an execution-based access layer over structured relations: a @LLM synthesizes an executable query (typically SQL) from a @NLQ which is then executed against a @DBMS to compute the relevant subset. In heterogeneous document settings, this has been motivated as a way to treat table-related subproblems as monolithic reasoning units, mitigating top-\(k\) chunk bias on global queries @yu-etal-2025-tablerag. The TAG framework generalizes this view, arguing that (i) pure Text2SQL only covers requests expressible in relational algebra and (ii) "row-retrieval @RAG" only supports point-lookup style questions; on the corresponding TAG-Bench, vanilla Text2SQL and vanilla RAG each plateau at roughly 20\% exact-match accuracy, whereas combined pipelines reach over 55\% @biswal2024text2sqlenoughunifyingai. These findings are consistent with the broader trend surfaced by large-scale benchmarks such as BIRD, which stresses dirty real-world databases, external knowledge requirements, and SQL efficiency, and on which even frontier models still fall well short of human performance @li2023bird.

While @LLM:pla have largely subsumed the role of earlier specialized neural parsers as the core SQL generation engine, state of the art systems still rely on orchestration layers that ensure the quality of the produced queries. We have identified four such techniques that are widely adopted across recent work:

+ *Schema linking.* A prerequisite for correct @SQL generation is schema linking. It ensures that mentions in the @NLQ align to the tables and columns they refer to in the database schema @hong2024nextgeneration @liu2024surveytexttosql. Errors at this stage cascade irreversibly: if the linker omits a required table or selects a wrong column, no downstream generation step can recover @cao2024rslsql. Early cross-domain systems relied on exact string matching between question tokens and schema item names, which breaks down as soon as synonyms or domain-specific paraphrases appear @wang2022proton. Graph neural network approaches addressed this by representing both the question and the schema as graphs and learning alignment through message passing, enabling the linker to exploit relational structure (e.g., foreign-key paths) rather than surface overlap alone @cai2021sadga. A subsequent line of work showed that pre-trained language models already encode usable schema-linking relations internally: probing these representations with unsupervised distance metrics yields robust linking even when surface forms diverge, without requiring additional parameters @wang2022proton. In @LLM\-based pipelines the schema-linking step is now typically implemented as a dedicated prompt or agent call that selects the relevant schema subset before the generation prompt is constructed @pourreza2023dinsql @gao2023texttosql.

+ *Execution-based verification and self-correction.* Early work on execution-guided decoding demonstrated that partially executing candidate programs against a @DBMS during beam search and pruning those that produce runtime failures like syntactic errors, semantic failures such as type mismatches and empty intermediate results, substantially improves output quality @wang2018executionguideddecoding. Modern @LLM pipelines adopt the same principle in a generate--execute--retry loop: the synthesized SQL is run against the database, and if it returns an error or an implausible result the @LLM is re-prompted with the error trace @gao2023texttosql @hong2024nextgeneration. This runtime feedback replaces the compile-time guarantees of earlier constrained-decoding methods that enforced syntactic validity by rejecting inadmissible tokens at each autoregressive step @scholak2021picard.

+ *Multi-agent decomposition.* Rather than relying on a single generation call, several recent systems distribute the task across specialised agents: one for schema linking, one for SQL drafting, one for refinement, and one for final selection. This allows each stage to be independently prompted and evaluated @li2025omnisql @pourreza2023dinsql.

+ *Retrieval-augmented example selection.* Retrieving semantically similar (question, @SQL) pairs from a curated store and injecting them as few-shot demonstrations has become a standard prompt-construction strategy, with reported execution accuracy of 86.6% on the Spider benchmark attributed in large part to this component @gao2023texttosql @yu2018spider.

Taken together, these techniques show that while the @LLM provides the core translation capability, robust Text2SQL still requires an engineered pipeline around it. However the nature of the scaffolding has shifted from grammar-level constraints to runtime orchestration.

== RAG on Knowledge Graphs

Prior work has also explored @RAG\-style systems grounded in knowledge graphs. When entity descriptions are serialized into text and processed through a standard chunking pipeline, the explicit relational structure of the graph is lost: edge types, multi-hop paths, and typed constraints cannot be recovered from unstructured text chunks @biswal2024text2sqlenoughunifyingai. This limitation is especially acute for Knowledge Graph Question Answering (KGQA) tasks that require compositional reasoning over graph topology rather than point-lookup of isolated facts.

To preserve relational structure, a growing body of work adopts a graph traversal-based paradigm analogous to Text2SQL: an @LLM synthesizes a @SPARQL query from a @NLQ, which is then executed against an RDF endpoint to retrieve the answer. However, Text2SPARQL presents distinct challenges relative to Text2SQL. Knowledge graph schemata are typically larger, more heterogeneous, and less standardized than relational database schemata, and the open-ended nature of RDF vocabularies means that entity and predicate identifiers often lack the mnemonic column names that aid SQL generation @kosten2023spider4sparql @kosten2025promptengineering. On the Spider4SPARQL benchmark, which transposed the complexity tiers of the original Spider benchmark into the @SPARQL domain, even frontier @LLM:pla achieve only up to 51\% execution accuracy, considerably below comparable Text2SQL results on the original Spider @kosten2023spider4sparql @kosten2025promptengineering. This gap confirms that KGQA remains substantially harder than its relational counterpart.

As with Text2SQL, state-of-the-art Text2SPARQL systems rely on orchestration layers around the core @LLM generation step. We identify four widely adopted techniques that parallel those in the relational setting:

+ *Ontology linking.* Domain-specific KGs with curated schemas such as PrimeKG (10 entity types, 18 relation types) used in the STaRK benchmark @stark2024 present vocabularies comparable in size and regularity to relational databases. For such smaller ontologies, a practical and widely adopted strategy is to inject the full ontology description directly into the @LLM system prompt. Reif et al. embed the complete ontology together with domain-specific standards into the prompt context, enabling accurate @SPARQL generation over industrial KGs without any fine-tuning @reif2024chatbotontology. Rasheed and Aguado show that augmenting prompts with discrete vocabulary information extracted from a reduced KG ontology yields competitive performance while substantially reducing prompt size, and that this approach enables off-domain users to query domain-specific KGs through a domain-agnostic interface @rasheed2025domainspecific. Hernandez-Camero et al. construct prompts that combine the natural-language question with contextual KG information (entity types, relation types, and example triples) alongside few-shot examples, improving @SPARQL accuracy by 6\% on an aviation KG @hernandezcamero2025aviation. This ontology-in-prompt strategy is effective when the schema fits within the @LLM context window, but does not scale to open-domain KGs with thousands of predicates, where more sophisticated ontology linking methods such as graph neural network alignment @cai2021sadga or learned hybrid prompts @jiang2025ontoscprompt become necessary.

+ *Execution-based verification and self-correction.* The generate--execute--retry loop transfers directly to the @SPARQL setting. FIRESPARQL incorporates a @SPARQL query correction layer that validates generated queries and repairs structural and semantic errors, achieving 0.90 ROUGE-L for query accuracy on the SciQA benchmark @pan2025firesparql. The Expasy federated @SPARQL system similarly includes a validation step that corrects generated queries against the KG endpoint, reducing hallucinated predicates and malformed triple patterns @emonet2024federatedsparql. GRASP goes further by using the @LLM to iteratively explore the knowledge graph through strategic @SPARQL sub-queries, discovering relevant IRIs and literals at runtime rather than relying on a static schema description. This agentic exploration achieves state-of-the-art results on multiple Wikidata benchmarks in a zero-shot setting @walter2025grasp.

+ *Retrieval-augmented example selection.* As in Text2SQL, retrieving semantically similar (question, @SPARQL) pairs for few-shot prompting is a standard strategy. SparqLLM retrieves template-based @SPARQL examples from a curated store to guide the @LLM toward structurally correct queries @arazzi2025sparqllm. Avila et al. combine @RAG with few-shot learning, retrieving both query examples and minimal KG subgraphs as context, achieving an F1-score of 0.73 in a zero-shot setting on SciQA which is a significant improvement over the prior score of 0.26 @avila2025fewshotrag. Kosten et al. systematically evaluate six prompting strategies on Spider4SPARQL and find that a simple prompt combined with an ontology description and five random shots is the most effective configuration @kosten2025promptengineering.

+ *Entity anchoring.* Before a multi-hop graph traversal can begin, the system must identify one or more _anchor entities_ --- KG nodes that correspond to the key concepts mentioned in the user query and serve as entry points into the graph. Shen et al. propose GeAR, which augments a conventional base retriever with a graph expansion mechanism that extracts proximal triples from initially retrieved nodes and maintains a gist memory across reasoning steps, achieving state-of-the-art results on multi-hop QA benchmarks such as MuSiQue while consuming fewer tokens than prior multi-step systems @shen2024gear. Xu et al. observe that most KG-based @RAG methods assume anchor entities are given, which breaks down in open-world settings where query terms do not map cleanly to KG nodes. Their AnchorRAG framework addresses this with a predictor agent that dynamically identifies candidate anchors by aligning query terms with KG nodes and spawns independent retriever agents for parallel multi-hop exploration from each candidate @xu2025anchorrag. Robust entity anchoring is particularly important for biomedical KGs such as PrimeKG, where synonymy, abbreviations, and cross-ontology identifiers make surface-level matching unreliable.

Taken together, these techniques demonstrate that while Text2SPARQL shares many orchestration patterns with Text2SQL, it faces additional challenges from the heterogeneity and scale of KG vocabularies and the absence of standardized schema conventions.

== Agentic RAG

Traditional @RAG pipelines follow a fixed retrieve-then-generate workflow: a query is embedded, the top-$k$ passages are fetched, and a single @LLM call synthesizes the answer. This static design lacks the adaptability required for multi-step reasoning, heterogeneous data sources, and queries whose complexity is not known in advance @singh2025agenticretrievalaugmentedgenerationsurvey. @aRAG addresses these limitations by embedding autonomous agents into the retrieval loop. Singh et al. identify four core agentic design patterns (reflection, planning, tool use, and multi-agent collaboration) that together allow the system to dynamically decide _what_ to retrieve, _how_ to retrieve it, and _when_ to stop @singh2025agenticretrievalaugmentedgenerationsurvey. Sapkota et al. draw a broader distinction between task-specific AI agents and full agentic AI systems, characterizing the latter by persistent memory, dynamic task decomposition, and coordinated autonomy across multiple specialized agents @sapkota2025aiagents.

Recent systems instantiate these patterns in different ways. HM-RAG employs a three-tiered hierarchy: a decomposition agent rewrites the query into coherent sub-tasks, modality-specific retrieval agents search vector, graph, and web sources in parallel, and a decision agent fuses answers through consistency voting, yielding a 12.95\% accuracy improvement on multimodal QA benchmarks @liu2025hmrag. MAO-ARAG takes an adaptive approach, training a planner agent via reinforcement learning to compose per-query workflows from a pool of executor agents (query reformulation, document selection, generation), balancing answer quality against cost and latency @chen2025maoarag.

Despite this progress, existing agentic systems typically treat SQL and graph backends as separate, independently managed tools and do not provide controlled experiments comparing when a graph query outperforms a relational query, or how an agent should decide between them. Our work addresses this gap.


= Approach <sec:approach>

#figure(
  placement: top,
  scope: "parent",
  image("agent_loop.png", width: 100%),
  caption: [Agent loop and system components.],
) <fig:agent-loop>

We present an agentic retrieval system for the STaRK-Prime knowledge base @stark2024, a biomedical knowledge graph derived from PrimeKG @chandak2023primekg containing over 129,000 entities across 10 types (diseases, drugs, genes/proteins, pathways, biological processes, molecular functions, cellular components, anatomical structures, exposures, and phenotypes) connected by 18 relation types. Our architecture combines semantic vector search with schema-bound query execution, enabling systematic comparison of retrieval strategies: pure semantic search versus structured querying via @SQL and @SPARQL.

== System Architecture <sec:architecture>

Our system follows the @aRAG paradigm, employing an autonomous agent loop as the core reasoning mechanism (see @fig:agent-loop). The agent iteratively reasons about the user's query, selects appropriate tools, executes them, and refines its approach based on intermediate results. This loop continues until the agent determines it has gathered sufficient information to answer the query or reaches a predefined iteration limit of 15 steps, which serves as a safeguard against infinite loops when the agent repeatedly fails to find relevant entities or formulates unsuccessful queries.

The architecture comprises three components:

+ *Agent Component*: An @LLM#[-driven] reasoning loop that interprets queries, plans tool invocations, and synthesizes final answers. We implement this using LangChain's ReAct agent framework. LangChain is the most widely adopted agent orchestration library in the research community, offering mature, well-documented abstractions for tool-using agents, composable chains, and callback-based observability @singh2025agenticretrievalaugmentedgenerationsurvey @liu2025hmrag @chen2025maoarag. Alternative frameworks pursue different design priorities: AutoGen @wu2023autogen centers on multi-agent _conversation_, where customizable agents communicate via natural-language message passing. This paradigm is well suited for collaborative reasoning but heavier than needed for our single-agent, multi-tool setting. AutoAgent @tang2025autoagent targets zero-code agent creation through natural language alone, prioritizing accessibility over fine-grained control of the retrieval pipeline. LangChain's explicit tool-calling API and its native integration with observability platforms such as Langfuse make it the most suitable choice for our controlled benchmarking setup, where reproducibility and per-step traceability are essential.

+ *Tool Component*: A set of specialized tools that provide the agent with distinct capabilities for knowledge base access. These tools abstract the complexity of different query interfaces behind a unified function-calling API.

+ *Data Component*: Multiple materialized views of the same underlying knowledge graph, each optimized for different query patterns.

== Two-Stage Query Process <sec:two-stage>

Prior work on the STaRK benchmark @stark2024 evaluated retrieval methods that operate in a single stage: given a natural language query, retrieve the top-$k$ most similar entities based on embedding distance or lexical matching. This approach struggles with multi-hop relational queries because entity descriptions alone do not encode graph structure.

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

*Use case*: Evaluating whether multi-paradigm structured query access yields complementary benefits.

== Model Selection <sec:model-selection>

We instantiate each agent configuration with two @LLM backbones to isolate the effect of model capacity:

- *GPT-5-mini*: A smaller, faster variant optimized for efficiency. Lower cost and latency make it suitable for high-throughput applications.
- *GPT-5*: The full-scale model with enhanced reasoning capabilities. Higher cost but potentially better performance on complex queries.

Both models are accessed via the OpenAI API with temperature set to 0.0 for near-deterministic output (note that API-level non-determinism from batching and floating-point arithmetic may still introduce minor variation across runs). We focus on the GPT family to control for API differences and prompting conventions, enabling cleaner isolation of model scale effects.

== Observability and Analysis <sec:observability>

To enable comprehensive analysis of agent behavior, we integrate observability and benchmarking infrastructure:

- *Langfuse*: Provides trace-level observability of agent reasoning, including tool invocations, intermediate results, and latency breakdowns. Each query execution is logged as a trace with full message history.

- *Custom Benchmark Framework*: Our evaluation harness executes agents across dataset splits, captures per-query metrics (precision, recall, F1, latency, tool call counts), and generates detailed analysis reports with item-level breakdowns and failure mode classification.

This instrumentation enables both real-time debugging during development and systematic post-hoc analysis of experimental results.



= Evaluation <sec:evaluation>

== Experimental Setup <sec:experimental-setup>

*Dataset.*
We evaluate on the STaRK Human-Generated QA benchmark @stark2024, comprising 109 natural-language questions over PrimeKG @chandak2023primekg, a biomedical knowledge graph. Questions require entity retrieval from a heterogeneous graph containing diseases, drugs, genes/proteins, pathways, biological processes, molecular functions, cellular components, anatomical structures, exposures, and phenotypes. Unlike synthetic benchmarks, the human-generated queries exhibit diverse linguistic patterns and varying complexity levels, including simple entity lookups, multi-hop relational queries, and complex constraint satisfaction problems.

*Agent Configurations.*
We compare four agentic retrieval strategies as described in @sec:agent-configurations: Search-Only, Search+SPARQL, Search+SQL, and Search+SQL+SPARQL. Each configuration provides access to different tool combinations, enabling systematic comparison of retrieval strategies. All agents use semantic search; Search-Only relies on it exclusively, while the others add structured query capabilities on top.

*Models.*
Following @sec:model-selection, each agent configuration is instantiated with two model variants: *GPT-5-mini* (smaller and faster) and *GPT-5* (full-scale). This pairing isolates the effect of model capacity on agentic retrieval performance.

*Repetitions.*
To quantify variance from @LLM non-determinism, each configuration-model pair is executed 6 times ($n=6$), yielding 48 total experimental runs and 5,232 individual query evaluations.

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

    [AvaTaR (GPT-4-turbo)], [33.0%], [51.4%], [53.3%], [41.0%],
    [Claude3 Reranker], [28.6%], [46.9%], [41.6%], [36.3%],
    [GPT4 Reranker], [28.6%], [44.9%], [41.6%], [34.8%],
    [GritLM-7b], [25.5%], [41.8%], [48.1%], [34.3%],
    [multi-ada-002], [24.5%], [39.8%], [47.2%], [33.0%],
    [BM25], [22.5%], [41.8%], [42.3%], [30.4%],

    table.hline(stroke: 1.5pt),
  ),
  caption: [STaRK baseline results on Human QA benchmark @stark2024. R\@20 = Recall at rank 20.],
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
    [28.1% #text(size: 0.8em)[(1.4)]],
    [30.9% #text(size: 0.8em)[(1.8)]],
    [24.8% #text(size: 0.8em)[(1.4)]],
    [11.2% #text(size: 0.8em)[(1.5)]],
    [34.1% #text(size: 0.8em)[(1.7)]],
    [42.0% #text(size: 0.8em)[(1.8)]],
    [37.2% #text(size: 0.8em)[(1.4)]],
    [23.8],

    [], [GPT-5],
    [37.9% #text(size: 0.8em)[(2.1)]],
    [38.2% #text(size: 0.8em)[(2.3)]],
    [32.9% #text(size: 0.8em)[(2.1)]],
    [17.6% #text(size: 0.8em)[(1.5)]],
    [40.3% #text(size: 0.8em)[(2.2)]],
    [51.2% #text(size: 0.8em)[(2.4)]],
    [44.5% #text(size: 0.8em)[(2.0)]],
    [38.6],

    table.hline(stroke: 0.5pt),

    // Search+SPARQL
    [Search+SPARQL], [GPT-5-mini],
    [46.2% #text(size: 0.8em)[(1.0)]],
    [47.2% #text(size: 0.8em)[(1.8)]],
    [40.3% #text(size: 0.8em)[(1.1)]],
    [24.3% #text(size: 0.8em)[(1.7)]],
    [46.7% #text(size: 0.8em)[(2.2)]],
    [56.7% #text(size: 0.8em)[(1.5)]],
    [50.6% #text(size: 0.8em)[(1.9)]],
    [42.5],

    [], [GPT-5],
    [#underline[47.7%] #text(size: 0.8em)[(1.9)]],
    [#underline[47.7%] #text(size: 0.8em)[(1.9)]],
    [#underline[40.4%] #text(size: 0.8em)[(1.8)]],
    [#underline[24.8%] #text(size: 0.8em)[(1.4)]],
    [*48.5%* #text(size: 0.8em)[(1.8)]],
    [#underline[58.4%] #text(size: 0.8em)[(3.4)]],
    [#underline[52.2%] #text(size: 0.8em)[(2.4)]],
    [64.4],

    table.hline(stroke: 0.5pt),

    // Search+SQL
    [Search+SQL], [GPT-5-mini],
    [44.3% #text(size: 0.8em)[(1.6)]],
    [46.5% #text(size: 0.8em)[(1.3)]],
    [38.3% #text(size: 0.8em)[(1.2)]],
    [20.2% #text(size: 0.8em)[(1.6)]],
    [44.9% #text(size: 0.8em)[(1.7)]],
    [57.8% #text(size: 0.8em)[(1.7)]],
    [49.9% #text(size: 0.8em)[(1.4)]],
    [47.6],

    [], [GPT-5],
    [45.3% #text(size: 0.8em)[(1.9)]],
    [42.5% #text(size: 0.8em)[(2.6)]],
    [37.1% #text(size: 0.8em)[(1.8)]],
    [22.8% #text(size: 0.8em)[(1.7)]],
    [46.8% #text(size: 0.8em)[(2.0)]],
    [54.1% #text(size: 0.8em)[(1.7)]],
    [49.9% #text(size: 0.8em)[(1.5)]],
    [70.9],

    table.hline(stroke: 0.5pt),

    // Search+SQL+SPARQL
    [Search+SQL+SPARQL], [GPT-5-mini],
    [*46.4%* #text(size: 0.8em)[(0.8)]],
    [*47.0%* #text(size: 0.8em)[(1.9)]],
    [39.4% #text(size: 0.8em)[(1.1)]],
    [20.9% #text(size: 0.8em)[(1.3)]],
    [#underline[46.9%] #text(size: 0.8em)[(0.9)]],
    [*59.8%* #text(size: 0.8em)[(2.4)]],
    [*52.1%* #text(size: 0.8em)[(0.9)]],
    [39.4],

    [], [GPT-5],
    [45.9% #text(size: 0.8em)[(2.4)]],
    [43.8% #text(size: 0.8em)[(2.0)]],
    [37.8% #text(size: 0.8em)[(1.9)]],
    [*23.2%* #text(size: 0.8em)[(2.1)]],
    [#underline[48.6%] #text(size: 0.8em)[(2.2)]],
    [56.1% #text(size: 0.8em)[(2.4)]],
    [51.8% #text(size: 0.8em)[(2.0)]],
    [76.1],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Our agentic retrieval performance on STaRK Human QA benchmark (mean over $n=6$ runs; std in parentheses). Best results per metric are *bolded*; second-best are #underline[underlined].],
) <tab:main-results>

== Analysis by Retrieval Strategy <sec:analysis>

=== Comparison to STaRK Baselines

Our agentic approaches substantially outperform all STaRK baselines. The best baseline, AvaTaR (GPT-4-turbo), achieves Hit\@1=33.0%, Hit\@5=51.4%, and MRR=41.0%. In comparison, our Search+SPARQL agent with GPT-5 achieves Hit\@1=48.5% (+15.5 pp), Hit\@5=58.4% (+7.0 pp), and MRR=52.2% (+11.2 pp).

We note that STaRK baselines also report Recall\@20 (R\@20), ranging from 41.6\% to 53.3\%. This metric is not directly comparable to our set-based Recall because STaRK baselines return fixed-size ranked lists of 20 candidates, whereas our agents return variable-sized answer sets (typically 1--5 entities); we therefore restrict the comparison to Hit\@$k$ and MRR.

This improvement is notable because STaRK baselines include strong retrieval methods: AvaTaR (an agentic retrieval approach), Claude3 and GPT-4 rerankers, state-of-the-art dense retrievers (GritLM-7b), and OpenAI embeddings (multi-ada-002). Our agentic approach with structured query capabilities provides a qualitative advantage over these methods. However, part of the improvement may be attributable to the stronger base model (GPT-5 / GPT-5-mini vs.\ GPT-4-turbo and earlier models used in STaRK baselines); a fully controlled comparison would require running all configurations on identical model backbones.

=== Structured vs. Unstructured Retrieval

The most striking result is the substantial performance gap between pure semantic search and structured-query agents. With GPT-5-mini, Search+SPARQL achieves:

- +18.1 pp higher precision (46.2% vs. 28.1%)
- +16.3 pp higher recall (47.2% vs. 30.9%)
- +15.5 pp higher F1 (40.3% vs. 24.8%)
- +13.1 pp higher exact match rate (24.3% vs. 11.2%)

This gap narrows but persists with GPT-5 (F1: 40.4% vs. 32.9%, +7.5 pp). The pattern confirms STaRK's original findings @stark2024: multi-hop relational queries such as "find diseases treated by drug X that also exhibit phenotype Y" exceed the capabilities of embedding-based retrieval. Semantic search over isolated entity descriptions cannot capture the graph structure needed for relational composition.

=== SPARQL vs. SQL

Search+SPARQL slightly outperforms Search+SQL across most metrics. With GPT-5-mini, SPARQL achieves +2.0 pp higher F1 (40.3% vs. 38.3%). The advantage persists with GPT-5 (+3.3 pp F1). While consistent, the gap is modest, suggesting that modern LLMs can effectively translate graph queries into relational operations.

=== Multi-Tool Agents

The Search+SQL+SPARQL agent achieves the highest Hit\@5 (59.8% with GPT-5-mini), outperforming both single-structured-query agents. However, its precision and F1 do not consistently exceed Search+SPARQL, and exact match rates are lower. This pattern suggests that multi-tool access improves recall by enabling the agent to recover entities that one query language might miss, but introduces additional decision complexity that can degrade precision.

=== Item-Level Correlation <sec:correlation>

To understand whether SPARQL and SQL provide complementary or redundant capabilities, we compute per-item F1 correlations between agent configurations (@tab:correlation). High correlation indicates that configurations succeed and fail on the same items.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, center, center, center, center),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Agent*], [*Search-Only*], [*SPARQL*], [*SQL*], [*SQL+SPARQL*],
    table.hline(stroke: 0.75pt),

    [Search-Only], [1.00], [.64/.40], [.69/.40], [.66/.37],
    [Search+SPARQL], [---], [1.00], [*.92/.90*], [.91/.93],
    [Search+SQL], [---], [---], [1.00], [.96/.95],
    [Search+SQL+SPARQL], [---], [---], [---], [1.00],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Per-item F1 correlation matrix (GPT-5/GPT-5-mini). Values show Pearson $r$ of item-level F1 scores averaged over 6 runs.],
) <tab:correlation>

The correlation between Search+SPARQL and Search+SQL is strikingly high: $r = 0.92$ (GPT-5) and $r = 0.90$ (GPT-5-mini). This indicates that the two query languages succeed and fail on nearly identical items, meaning they are *redundant* rather than complementary. By contrast, Search-Only shows much lower correlation with structured-query agents ($r = 0.40$ to $0.69$), confirming that structured queries provide qualitatively different capabilities than pure semantic search.

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

== Error Taxonomy <sec:error-taxonomy>

From detailed item-level analysis across all 109 benchmark queries and 48 experimental runs (5,232 query evaluations), we identify four primary failure modes with quantified incidence rates.

=== Entity Resolution Failure

The semantic search does not surface the correct entity, typically because:
- The entity has an unusual or ambiguous name.
- The query mentions the entity indirectly via properties rather than name.
- Multiple entities share similar names, and the wrong one is selected.

This is the most common failure mode: *34.9%* of queries (38/109) exhibit complete entity resolution failure (coverage = 0), while an additional *22.9%* (25/109) achieve only partial coverage. Only *42.2%* of queries achieve full coverage of gold entities during tool execution. Entity resolution failure is the dominant cause of the *38.5%* zero-recall rate observed across the benchmark. Notably, the semantic search tool itself exhibits *0% error rate* across all configurations: it reliably returns results, but those results may not contain the target entities.

=== Tool Execution Errors

@tab:tool-errors presents tool-level error rates across agent configurations. We distinguish between two metrics: _error rate per call_ (fraction of tool invocations that fail with an exception) and _items affected_ (fraction of queries experiencing at least one tool error).

#figure(
  table(
    columns: (auto, auto, 1fr, 1fr),
    align: (left, left, center, center),
    stroke: none,

    table.hline(stroke: 1.5pt),
    [*Agent*], [*Model*], [*Error Rate*], [*Items Affected*],
    table.hline(stroke: 0.75pt),

    [Search-Only], [GPT-5-mini], [0.00%], [0.0%],
    [], [GPT-5], [0.00%], [0.0%],
    table.hline(stroke: 0.5pt),

    [Search+SPARQL], [GPT-5-mini], [0.39%], [2.0%],
    [], [GPT-5], [0.53%], [3.8%],
    table.hline(stroke: 0.5pt),

    [Search+SQL], [GPT-5-mini], [1.06%], [5.8%],
    [], [GPT-5], [0.96%], [6.1%],
    table.hline(stroke: 0.5pt),

    [Search+SQL+SPARQL], [GPT-5-mini], [0.79%], [4.4%],
    [], [GPT-5], [0.64%], [4.3%],

    table.hline(stroke: 1.5pt),
  ),
  caption: [Tool execution error rates by agent configuration. Error Rate = errors/total calls. Items Affected = queries with $>=$ 1 error.],
) <tab:tool-errors>

Key observations:
- *Search tools are error-free*: Semantic search never fails at the API level, making Search-Only agents operationally robust.
- *SQL exhibits higher error rates than SPARQL*: Search+SQL agents experience 1.0% error rate vs. 0.4--0.5% for Search+SPARQL, with errors affecting 5.8--6.1% of items vs. 2.0--3.8%. SPARQL errors are exclusively syntax errors (malformed queries), whereas SQL errors are predominantly schema reference errors (invalid table or column names), indicating LLM confusion about the relational schema.
- *Model scale has limited impact on error rates*: GPT-5 does not substantially reduce error rates for SQL agents compared to GPT-5-mini, though SPARQL items affected nearly doubles from 2.0\% to 3.8\%, suggesting that the larger model may attempt more complex graph patterns that are harder to formulate correctly.

=== Zero-Result Queries (Semantic Failures)

Beyond execution errors, queries may succeed syntactically but return empty result sets, indicating semantically incorrect formulations. @tab:zero-results shows zero-result rates by tool type.

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

= Discussion <sec:discussion>

We now interpret the experimental results and derive practical recommendations for Agentic-RAG system design.

== Structured Query Generation as the Key Advantage <sec:discussion-structured>

The dominant finding is that agentic access to structured query interfaces (SPARQL or SQL) substantially outperforms both pure semantic retrieval and all STaRK baselines (@tab:main-results, @tab:stark-baselines). The +15.5 pp F1 improvement of Search+SPARQL over Search-Only (with GPT-5-mini) represents a qualitative capability gap, not merely incremental improvement.

This confirms and extends the central finding from the STaRK benchmark @stark2024: embedding-based retrieval struggles with multi-hop relational queries that require joins across entity types, path traversals, or constraint filtering. The agentic paradigm---combining iterative reasoning with structured query execution---provides a path forward that pure retrieval methods cannot match.

Importantly, this advantage emerges even though all structured-query agents _depend on_ semantic search for entity resolution. The structured query step adds value by enabling explicit relational reasoning over the resolved entities, whereas Search-Only terminates after the initial retrieval step. However, as detailed in @sec:error-taxonomy, our analysis also reveals *missed opportunities*---a reasoning failure at answer synthesis distinct from retrieval failure.

== The Entity Resolution Bottleneck <sec:discussion-bottleneck>

A critical architectural insight from item-level analysis is that entity resolution constitutes the primary failure mode for structured-query agents. The pipeline operates as follows:

+ Semantic search maps natural-language entity mentions to node IDs.
+ The LLM formulates a structured query using these IDs.
+ Query execution retrieves related entities from the graph.

If step (1) fails to surface the correct entity, no subsequent query can recover. We observe cases where agents execute 10+ tool calls across 9 iterations, generating syntactically valid queries, yet fail because the initial entity search returned incorrect node IDs. For example, item 3 in our benchmark ("gene involved in vesicle transport, located in kinetochore, in antigen processing pathway") required finding entity ID 5585, which never appeared in any of 11 search results despite extensive query reformulation.

This bottleneck suggests that improving entity resolution quality---via better embeddings, synonym expansion, fuzzy matching, or hybrid lexical-semantic retrieval---may yield larger gains than improving LLM reasoning capacity for query generation.

== Diminishing Returns from Model Scale <sec:discussion-scaling>

The observation that GPT-5 substantially improves Search-Only but yields marginal or no improvement for structured-query agents is theoretically significant. We hypothesize two contributing factors:

+ *Query generation saturation*: GPT-5-mini already generates correct SPARQL/SQL for most queries. Additional reasoning capacity provides diminishing returns.
+ *Bottleneck shift*: With structured queries, performance is limited by entity resolution and knowledge graph coverage---factors orthogonal to LLM scale.

This finding aligns with recent text-to-SQL research showing that smaller models can achieve competitive accuracy when provided with adequate schema context @mohammadjafari2025naturallanguagesqlreview. From a practical standpoint, smaller models are preferable for structured-query agents, reserving larger models for pure semantic retrieval where their advantages are realized.

== Multi-Tool Agents: Redundancy over Complementarity <sec:discussion-multitool>

As noted in @sec:analysis, Search+SQL+SPARQL agents achieve the highest Hit\@5 but do not dominate on precision or F1. This is surprising: one might expect that providing access to both a graph-native query language (SPARQL) and a relational query language (SQL) would yield complementary benefits, with each excelling on different query types.

However, the item-level correlation analysis (@sec:correlation, @tab:correlation) reveals that SQL and SPARQL are largely *redundant* rather than complementary. The per-item F1 correlation between Search+SPARQL and Search+SQL is $r = 0.92$ (GPT-5) and $r = 0.90$ (GPT-5-mini)---indicating that these configurations succeed and fail on nearly identical items. Only 12.8% of items show complementary behavior (one language succeeds while the other fails); the remaining 87.2% are redundant (both succeed or both fail).

When both tools are available, the agent does not consistently leverage one for graph patterns and the other for aggregations---instead, it faces additional decision overhead without gaining new expressive power. We attribute the multi-tool agent's failure to dominate to three factors:

- *Tool selection overhead*: Choosing between SPARQL and SQL introduces an additional decision point that can lead to suboptimal selections. Notably, when given both options, the agent uses SQL exclusively 59% of the time despite SPARQL achieving higher overall F1.
- *Redundant tool calls*: Agents sometimes invoke both query tools on the same subproblem, adding latency without improving recall.
- *Prompt complexity*: More tool options increase prompt length and complexity, potentially degrading query formulation quality.

These findings suggest that practitioners should choose *one* structured query language based on infrastructure constraints rather than providing both. Multi-tool agents may still benefit recall-critical applications where exhaustive entity discovery outweighs precision, but the expected complementarity between SQL and SPARQL does not materialize in practice.

== SPARQL vs. SQL: Practical Equivalence <sec:discussion-sparql-sql>

As shown in @sec:analysis, SPARQL's advantage over SQL is consistent but surprisingly small (+2--3 pp F1). The high item-level correlation ($r > 0.9$, @tab:correlation) suggests that the two languages succeed and fail on largely the same items, indicating broad functional overlap for knowledge graph retrieval. However, the languages are not fully equivalent in operational characteristics: SQL exhibits roughly 2--3$times$ higher error rates and substantially higher zero-result rates (50--63\% vs.\ 25--53\% for SPARQL), indicating that LLMs struggle more with relational query formulation over projected graph schemas. These differences have practical deployment implications:

- Organizations with existing SQL infrastructure can achieve competitive retrieval quality without deploying dedicated SPARQL endpoints, though they should expect somewhat higher query failure rates.
- SPARQL's lower error and zero-result rates make it the safer default when both options are available.
- The choice between SPARQL and SQL should be driven primarily by infrastructure constraints and developer familiarity, with SPARQL preferred when no strong infrastructure reason favors SQL.

== Design Recommendations for Agentic-RAG Systems <sec:discussion-recommendations>

Based on our findings, we offer the following recommendations for practitioners building Agentic-RAG systems over mix-typed knowledge bases:

+ *Always include structured query access* when the knowledge base supports it. The performance gap over pure semantic search is large (+15.5 pp F1) and consistent across model scales.

+ *Invest in entity resolution quality*. This is the primary bottleneck for structured-query agents. Consider:
  - Fine-tuning entity embeddings on domain-specific corpora.
  - Implementing synonym expansion and alias matching.
  - Hybrid lexical-semantic retrieval (e.g., BM25 + dense retrieval fusion).

+ *Use smaller models for structured-query agents*. GPT-5-mini achieves comparable or better F1 than GPT-5 at half the latency and cost.

+ *Avoid multi-tool agents*. Despite intuitive appeal, providing both SQL and SPARQL yields no significant benefit. The high item-level correlation ($r > 0.9$) indicates redundancy, and the combined agent's marginal Hit\@5 improvement (+2 pp) is not statistically significant given observed variance.

+ *Choose one structured query language*. SPARQL slightly outperforms SQL (+2--3 pp F1), but the difference is modest. Select based on infrastructure constraints and team familiarity rather than expected retrieval quality.

+ *Implement result validation*. The "missed opportunity" failure mode suggests that explicit re-ranking or verification of intermediate results could improve final answer quality.

== Limitations <sec:limitations>

Our study has several limitations that suggest directions for future work:

- *Single domain*: Evaluation is limited to the biomedical domain (PrimeKG). Results may differ on other knowledge graphs with different schema complexity, entity distributions, or query patterns.

- *Single benchmark*: We evaluate on STaRK Human QA ($n=109$). Larger-scale evaluation on diverse benchmarks would strengthen generalizability claims.

- *Single model family*: Only GPT-5 variants are compared. Open-source models (LLaMA, Mistral) or other commercial models (Claude, Gemini) may exhibit different performance characteristics.

- *Entity retrieval only*: We evaluate entity set retrieval, not end-to-end question answering with natural-language generation. Answer synthesis quality is an orthogonal concern.

- *Controlled concurrency*: Latency measurements use concurrency=5, which may not reflect production deployment conditions with varying load patterns.

- *Static knowledge graph*: We do not evaluate performance on evolving knowledge graphs or the impact of knowledge staleness.

- *Prompt sensitivity*: Our system prompt (@sec:system-prompt) is extensively engineered with retry limits, parallelization instructions, and output format constraints. We do not evaluate sensitivity to prompt phrasing; different formulations may yield different performance characteristics.

- *No statistical significance testing*: Despite running 6 repetitions per configuration, we report only means and standard deviations without formal significance tests. Differences of 2--3 pp between SPARQL and SQL fall within observed standard deviations, so the SPARQL advantage may not be statistically significant.

= Conclusion

We presented an agentic RAG system that combines semantic vector search with schema-bound query execution (SQL and SPARQL) over the STaRK-Prime biomedical knowledge graph. Through a controlled comparison of four agent configurations across two model scales and six repetitions, we established three main findings. First, structured query access provides a substantial advantage over pure semantic search (+15.5 pp F1), confirming that multi-hop relational queries exceed the capabilities of embedding-based retrieval. Second, SQL and SPARQL are largely redundant rather than complementary ($r > 0.9$ item-level correlation), with SPARQL holding a modest but consistent edge (+2--3 pp F1); practitioners should choose one query language based on infrastructure constraints rather than providing both. Third, the dominant bottleneck for structured-query agents is entity resolution, not LLM reasoning capacity: scaling from GPT-5-mini to GPT-5 yields minimal improvement when structured query tools are available, while 34.9% of queries fail due to incomplete entity coverage during semantic search.

These results suggest that future work should prioritize improving entity resolution quality---through better embeddings, synonym expansion, or hybrid lexical-semantic retrieval---over more sophisticated query generation or larger language models. Extending this evaluation to additional domains, knowledge graph schemas, and open-source model families would further strengthen the generalizability of our findings.

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

      *Result:* Precision=100%, Recall=100%, F1=100%, Latency=62s
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

      *Result:* Precision=0%, Recall=0%, F1=0%, Latency=146s

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

      *Result:* Precision=100%, Recall=50%, F1=67% | Missed opportunity rate: 50%

      *Analysis:* The agent discovered both gold entities (95312 via search, 35882 via SPARQL) but only returned the SPARQL result. Entity 95312 was found but not included in the final answer.
    ],
  ),
  caption: [Missed opportunity: agent discovers correct entity but fails to include it in final prediction.],
) <fig:example-missed>

This example illustrates the *incomplete result selection* failure mode:

+ *Full coverage achieved*: Both gold entities were discovered during tool execution (coverage = 100%).

+ *Filtering failure*: The agent correctly identified 95312 as the query entity but interpreted the question as asking only for the _parent_ disease, not the disease hierarchy. This represents a reasoning failure at the answer synthesis stage.

+ *Actionable improvement*: Explicit re-ranking or verification of intermediate results---checking whether discovered entities should be included in the final answer---could address this failure mode without improving entity resolution or query generation.

== A.4 Agent System Prompt <sec:system-prompt>

The following is the complete system prompt used for the Search+SQL+SPARQL agent configuration. This prompt is instantiated at runtime with the actual database schema and vocabulary information. The `{schema_and_vocab}` and `{max_rows}` placeholders are replaced at runtime with the actual database schema and query limits.

#block(
  breakable: true,
  width: 100%,
  inset: 8pt,
  stroke: 0.5pt + gray,
  radius: 4pt,
  fill: luma(250),
)[
  #set text(size: 6.5pt)
  #set par(justify: false)
  ```
  You are an expert biomedical knowledge base analyst. Your task is to answer
  questions about diseases, drugs, genes/proteins, pathways, molecular functions,
  and their relationships using the STaRK-Prime knowledge base.

  ## Available Tools

  You have access to THREE tools:

  1. **search_entities_tool** - Semantic search to find entities by name/description
     - Use this FIRST to resolve entity names to their IDs
     - Handles synonyms, partial matches, and related terms
     - Returns entity IDs that you can use in queries

  2. **execute_sql_query_tool** - Execute SQL queries (PostgreSQL)
      - Use AFTER finding entity IDs with search_entities_tool
      - Best for aggregations, joins, and filtering

  3. **execute_sparql_query_tool** - Execute SPARQL queries (Fuseki)
      - Use AFTER finding entity IDs with search_entities_tool
      - Best for path traversal and pattern matching

  ## Two-Stage Query Process (IMPORTANT!)

  **ALWAYS follow this two-stage approach:**

  ### Stage 1: Entity Resolution
  When a question mentions specific entities (diseases, drugs, genes, etc.):
  1. Use `search_entities_tool` to find the entity IDs
  2. Note the returned IDs for use in your query

  Example: For "What genes are associated with both diabetes and hypertension?"
  -> First: search_entities_tool("diabetes", "disease")
           search_entities_tool("hypertension", "disease")
  -> Get: diabetes ID 12345, hypertension ID 67890

  ### Stage 2: Query Execution
  Use the resolved entity IDs in your SQL or SPARQL query.

  ## Exploration Strategy

  ### Strategy 1: Queries without explicit entity mentions
  Use `search_entities_tool` with the full question as the query and `top_k=15`.

  ### Strategy 2: Queries with explicit entity mentions
  1. Resolve each entity with `search_entities_tool` (use `entity_type` when obvious).
  2. Proceed to query execution with the resolved IDs.

  ### Strategy 3: Multi-entity or complex queries
  1. Disambiguate all mentioned entities.
  2. Explore neighborhoods of key entities with relevant filters.
  3. Combine information from multiple exploration paths.

  ## Query Language Selection

  - **SQL** is better for: lookups, aggregations (COUNT, AVG, MAX), joins,
    complex filters, questions asking "how many" or "list all"
  - **SPARQL** is better for: path traversal, relationship exploration,
    pattern matching, questions like "what is connected to X through Y"

  ## Knowledge Base Schema

  {schema_and_vocab}

  ## Query Guidelines

  1. **Entity Resolution First**: ALWAYS use search_entities_tool to find entity
     IDs before querying. Do NOT try to match entity names with SQL LIKE or
     SPARQL FILTER - use semantic search instead.
  2. **Read-only only**: SQL must be SELECT-only; SPARQL must be read-only.
  3. **Limit results**: Always use LIMIT {max_rows}.
  4. **Be precise**: Use exact table/column names from the schema above.
  5. **Handle errors**: If a query fails, analyze the error and try again.
  6. **Node IDs are answers**: The benchmark expects node IDs (integers).

  ## Answer Format

  Output ONLY a JSON object with exactly two fields:
  {"ids": [123, 456, 789], "reasoning": "Found 3 genes associated with diabetes"}

  - The `ids` field must be an array of integers, empty array [] if no results
  - The `reasoning` field must be a string explaining your process
  - IDs preserve ranking order for Hit@1 and MRR metrics

  ## Efficiency Guidelines

  ### Parallelization
  - Resolve multiple entities IN PARALLEL in ONE turn (max 4-5 searches)

  ### Retry Limits
  - Query errors: max 2 corrective attempts per query
  - Zero-row results: max 2 query reformulations
  - Hard cap: 6 tool rounds total. If reached, answer with best available IDs

  ### Answer Strategy
  - If a query returns 0 rows, adjust and retry (within retry limits)
  - Partial results are valuable - report entity IDs even without relationship data

  ## Query Examples

  Example SQL workflow:
  1. search_entities_tool("breast cancer", "disease") -> ID: 789
  2. execute_sql_query_tool("SELECT dst_id FROM indication WHERE src_id = 789")

  Example SPARQL workflow:
  1. search_entities_tool("insulin", "gene_protein") -> ID: 456
  2. execute_sparql_query_tool("PREFIX sp: <http://stark.stanford.edu/prime/>
     SELECT ?related WHERE { sp:node/456 sp:associatedWith ?related } LIMIT 5")

  Now answer the user's question using the two-stage approach.
  ```
]
