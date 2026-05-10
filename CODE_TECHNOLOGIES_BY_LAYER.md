# Code Technologies by Layer and Node

## Pipeline Summary

- Codebase area reviewed: `code/src`
- Primary language: Python
- Orchestration: LangGraph `StateGraph`
- Total execution layers: 6
- Total execution nodes: 24
- Core architecture: layered legal RAG pipeline over Akoma Ntoso documents and an RDF knowledge graph
- LLM providers wired in code: Ollama, Groq, AI Grid

## Layer Summary

| Layer | Nodes | Main technologies |
| --- | ---: | --- |
| Layer 0 | 4 | `lxml`, `rdflib`, RDF/Turtle KG loading, Python dict indexing, temporal KG indexing |
| Layer 1 | 4 | Arabic text normalization, regex/rule-based query analysis, hybrid rule + LLM classification |
| Layer 2 | 6 | BM25, SPLADE, dense embeddings, FAISS, ColBERT-style late interaction, GraphRAG, SPARQL, temporal retrieval |
| Layer 3 | 2 | Reciprocal Rank Fusion, weighted hybrid fusion, cross-encoder reranking |
| Layer 4 | 3 | CRAG-style correction, Self-RAG scoring, corrective retrieval loop |
| Layer 5 | 5 | evidence packing, abstention gating, ADU extraction, LLM answer generation, citation auditing |

## Layer 0

| Node | File | Technologies used |
| --- | --- | --- |
| `parse_akn` | `code/src/layer0/akn_parser.py` | `lxml.etree`; Akoma Ntoso XML parsing; article and paragraph extraction; legal unit segmentation |
| `build_rdf_kg` | `code/src/layer0/rdf_kg.py` | `rdflib`; RDF/Turtle parsing; ontology namespaces; in-memory KG assembly |
| `build_citation_registry` | `code/src/layer0/citation_registry.py` | Python `dict`; citation-to-legal-unit registry; O(1) lookup |
| `temporal_index_node` | `code/src/temporal/temporal_index.py` | `lxml.etree`; temporal metadata parsing; amendment-chain indexing derived from KG events; cache serialization |

## Layer 1

| Node | File | Technologies used |
| --- | --- | --- |
| `arabic_normalize` | `code/src/layer1/arabic_normalizer.py` | Python `re`; Arabic orthographic normalization; diacritic stripping; whitespace normalization |
| `acqo_classify` | `code/src/layer1/acqo_classifier.py` | hybrid classifier; regex pre-router from `router_rules.py`; optional LLM classification through `llm_client`; default code-path model family `llama3.2` |
| `temporal_intent_node` | `code/src/temporal/intent_detector.py` | deterministic regex temporal-intent detection; year/date parsing; law-reference extraction |
| `router` | `code/src/layer1/router.py` | static routing tables; retriever weighting; query-type-specific plans; GraphRAG and temporal gating |

## Layer 2

| Node | File | Technologies used |
| --- | --- | --- |
| `bm25_retriever` | `code/src/layer2/bm25_retriever.py` | `rank_bm25.BM25Okapi`; lexical retrieval; Arabic-normalized tokenization; metadata-aware scoring |
| `splade_retriever` | `code/src/layer2/splade_retriever.py` | `sentence_transformers.SparseEncoder`; SPLADE sparse retrieval; `torch`; inverted index; query expansion; TF-IDF fallback |
| `dense_retriever` | `code/src/layer2/dense_retriever.py` | `sentence_transformers.SentenceTransformer`; FAISS `IndexFlatIP`; `torch`; `numpy`; multilingual embedding retrieval; default model `intfloat/multilingual-e5-large` |
| `colbert_retriever` | `code/src/layer2/colbert_retriever.py` | `transformers.AutoTokenizer`; `transformers.AutoModel`; `torch`; ColBERT-style MaxSim late interaction; default encoder `aubmindlab/bert-base-arabertv2` |
| `graphrag_retriever` | `code/src/layer2/graphrag_retriever.py` | `rdflib.Graph`; SPARQL; graph-path retrieval; KG grounding from `graphrag_grounding.py`; cached entity and provision lookup |
| `temporal_retriever` | `code/src/temporal/retriever.py` | version-aware temporal retrieval; amendment-chain scoring; date filtering; citation-registry lookup; lexical fallback over temporal chains |

## Layer 3

| Node | File | Technologies used |
| --- | --- | --- |
| `fusion_node` | `code/src/layer3/fusion.py` | Reciprocal Rank Fusion; weighted hybrid fusion; graph-boost strategy |
| `rerank_node` | `code/src/layer3/reranker.py` | `sentence_transformers.CrossEncoder`; `torch`; neural reranking; default model `nreimers/mmarco-mMiniLMv2-L12-H384-v1` |

## Layer 4

| Node | File | Technologies used |
| --- | --- | --- |
| `corrective_rag_node` | `code/src/layer4/corrective_rag.py` | CRAG-style retrieval quality scoring; threshold-based correction labels |
| `self_rag_node` | `code/src/layer4/self_rag.py` | heuristic relevance scoring; optional LLM self-reflection through `llm_client`; default self-reflection model family `llama3.2` |
| `corrective_loop_node` | `code/src/layer4/corrective_loop.py` | corrective query expansion; BM25 plus dense re-search; GraphRAG cross-reference hop recovery; article-level deduplication |

## Layer 5

| Node | File | Technologies used |
| --- | --- | --- |
| `evidence_packer_node` | `code/src/layer5/evidence_packer.py` | structured evidence packing; article-level aggregation; temporal prompt augmentation |
| `abstention_gate_node` | `code/src/layer5/abstention_gate.py` | regex-based out-of-scope and trap detection; answer gating |
| `adu_extractor_node` | `code/src/adu/extractor.py` | LLM-based ADU triple extraction; JSON validation in `adu/validator.py`; heuristic fallback; default model family `llama3` with `llama3.2` fallback |
| `generator_node` | `code/src/layer5/generator.py` | provider-agnostic chat generation via `llm_client`; citation-grounded prompting; route-specific model selection; default Ollama-targeted model families `llama3.2` and `llama3` |
| `citeguard_node` | `code/src/layer5/citeguard.py` | regex citation extraction; character-bigram Jaccard similarity; post-generation citation audit |

## Shared Services

| Module | File | Technologies used |
| --- | --- | --- |
| Graph orchestration | `code/src/graph.py` | LangGraph `StateGraph`; conditional edges; layered node pipeline |
| Shared state model | `code/src/state.py` | `dataclasses`; `TypedDict`; SHA-256-based citation IDs |
| LLM adapter | `code/src/llm_client.py` | `urllib.request`; JSON payload handling; provider adapters for Ollama and OpenAI-compatible chat APIs |
| Temporal KG event derivation | `code/src/temporal/kg_events.py` | `rdflib`; URI parsing; amendment-event extraction from RDF relations |
| Graph grounding | `code/src/layer2/graphrag_grounding.py` | regex; dataclasses; alias-based entity grounding from natural-language law references |
