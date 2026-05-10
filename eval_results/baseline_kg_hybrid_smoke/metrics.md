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
| citation_f1 | 0.1658 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.1125 |
| citation_recall | 0.3979 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4250 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1539 |
| map_doc | 0.4948 |
| mean_latency_s | 17.6802 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2229 |
| mrr_doc | 0.5365 |
| ndcg_article | 0.2847 |
| ndcg_doc | 0.6058 |
| precision_article | 0.0562 |
| precision_doc | 0.0813 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.3979 |
| recall_doc | 0.7500 |
| rouge_l | 0.0485 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0728 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6667 | 0.1000 | 0.1667 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.3929 | 0.0000 | 0.0000 |
| layman | 0.3750 | 0.1250 | 0.1667 | 0.0000 | 0.0000 |
| long_context | 0.5000 | 0.1000 | 0.1000 | 0.0000 | 0.0000 |
| multi_hop | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.3333 | 0.3333 | 0.3333 | 0.0000 | 0.0000 |
| unanswerable | 0.7500 | 0.1250 | 0.1667 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7500 | 0.7500 | 0.3730 | 0.0000 | 0.0000 |
| hard | 0.5625 | 0.1396 | 0.1500 | 0.0000 | 0.0000 |
| medium | 0.3667 | 0.0400 | 0.0667 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5365 | 0.2229 | 0.1658 | 0.0000 | 0.0000 |
