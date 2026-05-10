# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7254 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0929 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0680 |
| citation_recall | 0.1863 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3954 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0164 |
| map_article | 0.1297 |
| map_doc | 0.5399 |
| mean_latency_s | 0.0165 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1971 |
| mrr_doc | 0.5792 |
| ndcg_article | 0.2168 |
| ndcg_doc | 0.6226 |
| precision_article | 0.0328 |
| precision_doc | 0.0783 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1863 |
| recall_doc | 0.7008 |
| rouge_l | 0.0367 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0634 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5556 | 0.1250 | 0.0833 | 0.0000 | 0.0000 |
| exact_article | 0.6342 | 0.3282 | 0.1523 | 0.0000 | 0.0000 |
| layman | 0.3922 | 0.0196 | 0.0235 | 0.0000 | 0.0000 |
| long_context | 0.6765 | 0.2010 | 0.0738 | 0.0000 | 0.0000 |
| multi_hop | 0.7096 | 0.1212 | 0.0586 | 0.0000 | 0.0000 |
| rule_application | 0.6412 | 0.3040 | 0.1392 | 0.0000 | 0.0000 |
| temporal_factual | 0.5595 | 0.0357 | 0.0476 | 0.0000 | 0.0000 |
| unanswerable | 0.3600 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5595 | 0.2759 | 0.1287 | 0.0000 | 0.0000 |
| hard | 0.5625 | 0.1016 | 0.0467 | 0.0000 | 0.0000 |
| medium | 0.6125 | 0.2608 | 0.1254 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6092 | 0.2073 | 0.0978 | 0.0000 | 0.0000 |
| fr | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
