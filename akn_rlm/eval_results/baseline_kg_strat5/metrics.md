# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7500 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0050 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0050 |
| citation_recall | 0.0050 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.0625 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.0025 |
| map_doc | 0.0750 |
| mean_latency_s | 14.8779 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.0125 |
| mrr_doc | 0.0875 |
| ndcg_article | 0.0158 |
| ndcg_doc | 0.0973 |
| precision_article | 0.0025 |
| precision_doc | 0.0125 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.0050 |
| recall_doc | 0.1125 |
| rouge_l | 0.0260 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0445 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| layman | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.1000 | 0.1000 | 0.0400 | 0.0000 | 0.0000 |
| multi_hop | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hard | 0.0500 | 0.0250 | 0.0100 | 0.0000 | 0.0000 |
| medium | 0.1500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.0875 | 0.0125 | 0.0050 | 0.0000 | 0.0000 |
