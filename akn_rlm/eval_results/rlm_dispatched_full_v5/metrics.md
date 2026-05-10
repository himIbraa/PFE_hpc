# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8156 |
| abstention_f1 | 0.6980 |
| abstention_precision | 0.6341 |
| abstention_recall | 0.7761 |
| answer_faithfulness | 0.3361 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2960 |
| citation_groundedness | 0.5667 |
| citation_precision | 0.2034 |
| citation_recall | 0.2049 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5969 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.1865 |
| map_doc | 0.5058 |
| mean_latency_s | 4.3461 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2730 |
| mrr_doc | 0.5451 |
| ndcg_article | 0.2810 |
| ndcg_doc | 0.5504 |
| precision_article | 0.0340 |
| precision_doc | 0.0570 |
| reasoning_chain_score | 0.0055 |
| recall_article | 0.2049 |
| recall_doc | 0.5253 |
| rouge_l | 0.0947 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1178 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5417 | 0.1625 | 0.1071 | 0.0000 | 0.0000 |
| exact_article | 0.5932 | 0.4576 | 0.4192 | 0.4167 | 0.3448 |
| layman | 0.5882 | 0.1471 | 0.2824 | 1.0000 | 0.5000 |
| long_context | 0.8824 | 0.2010 | 0.0991 | 0.0000 | 0.0000 |
| multi_hop | 0.6538 | 0.1923 | 0.1053 | 0.6667 | 0.5000 |
| rule_application | 0.6667 | 0.3826 | 0.2218 | 0.4286 | 0.2857 |
| temporal_factual | 0.7857 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5893 | 0.3750 | 0.3435 | 0.4167 | 0.3571 |
| hard | 0.4612 | 0.1570 | 0.2601 | 0.9762 | 0.9318 |
| medium | 0.6176 | 0.3465 | 0.3081 | 0.4615 | 0.3636 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5647 | 0.2871 | 0.3069 | 0.7656 | 0.7206 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
