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
| citation_f1 | 0.0876 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0625 |
| citation_recall | 0.1896 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4417 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1427 |
| map_doc | 0.5573 |
| mean_latency_s | 1.2351 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2656 |
| mrr_doc | 0.6198 |
| ndcg_article | 0.2769 |
| ndcg_doc | 0.6827 |
| precision_article | 0.0312 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1896 |
| recall_doc | 0.8125 |
| rouge_l | 0.0433 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0708 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.2679 | 0.0000 | 0.0000 |
| layman | 1.0000 | 0.5000 | 0.1667 | 0.0000 | 0.0000 |
| long_context | 0.6250 | 0.5000 | 0.1000 | 0.0000 | 0.0000 |
| multi_hop | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.2500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unanswerable | 0.2500 | 0.1250 | 0.1667 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.2897 | 0.0000 | 0.0000 |
| hard | 0.5104 | 0.1562 | 0.0667 | 0.0000 | 0.0000 |
| medium | 0.5667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6198 | 0.2656 | 0.0876 | 0.0000 | 0.0000 |
