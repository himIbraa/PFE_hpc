# Evaluation Metrics

**Total questions:** 16

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8125 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0125 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0125 |
| citation_recall | 0.0125 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.1562 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.0063 |
| map_doc | 0.1562 |
| mean_latency_s | 14.4230 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.0312 |
| mrr_doc | 0.1562 |
| ndcg_article | 0.0394 |
| ndcg_doc | 0.1808 |
| precision_article | 0.0063 |
| precision_doc | 0.0250 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.0125 |
| recall_doc | 0.2500 |
| rouge_l | 0.0279 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0477 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layman | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.2500 | 0.2500 | 0.1000 | 0.0000 | 0.0000 |
| multi_hop | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hard | 0.1250 | 0.0625 | 0.0250 | 0.0000 | 0.0000 |
| medium | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.1562 | 0.0312 | 0.0125 | 0.0000 | 0.0000 |
