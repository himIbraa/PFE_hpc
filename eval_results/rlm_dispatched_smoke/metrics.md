# Evaluation Metrics

**Total questions:** 16

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.9375 |
| abstention_f1 | 0.8000 |
| abstention_precision | 1.0000 |
| abstention_recall | 0.6667 |
| answer_faithfulness | 0.1250 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2375 |
| citation_groundedness | 0.6771 |
| citation_precision | 0.2354 |
| citation_recall | 0.3271 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5312 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2529 |
| map_doc | 0.6250 |
| mean_latency_s | 10.7333 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.3563 |
| mrr_doc | 0.6562 |
| ndcg_article | 0.3707 |
| ndcg_doc | 0.6644 |
| precision_article | 0.0562 |
| precision_doc | 0.0687 |
| reasoning_chain_score | 0.0150 |
| recall_article | 0.3271 |
| recall_doc | 0.6562 |
| rouge_l | 0.1235 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1612 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.5333 | 0.0000 | 0.0000 |
| layman | 1.0000 | 1.0000 | 0.9000 | 0.0000 | 0.0000 |
| long_context | 1.0000 | 0.2500 | 0.1333 | 0.0000 | 0.0000 |
| multi_hop | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.7500 | 0.6000 | 0.3333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.6889 | 0.0000 | 0.0000 |
| hard | 0.5625 | 0.2125 | 0.1167 | 1.0000 | 1.0000 |
| medium | 0.6000 | 0.2000 | 0.1600 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6562 | 0.3563 | 0.2375 | 0.6667 | 0.8000 |
