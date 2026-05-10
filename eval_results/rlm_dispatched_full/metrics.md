# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8156 |
| abstention_f1 | 0.7020 |
| abstention_precision | 0.6310 |
| abstention_recall | 0.7910 |
| answer_faithfulness | 0.3443 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2929 |
| citation_groundedness | 0.5086 |
| citation_precision | 0.1828 |
| citation_recall | 0.2172 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5951 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1835 |
| map_doc | 0.4976 |
| mean_latency_s | 3.9178 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2569 |
| mrr_doc | 0.5232 |
| ndcg_article | 0.2710 |
| ndcg_doc | 0.5366 |
| precision_article | 0.0348 |
| precision_doc | 0.0594 |
| reasoning_chain_score | 0.0051 |
| recall_article | 0.2172 |
| recall_doc | 0.5444 |
| rouge_l | 0.0914 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1156 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.1208 | 0.1071 | 0.0000 | 0.0000 |
| exact_article | 0.6186 | 0.4633 | 0.4111 | 0.4167 | 0.3571 |
| layman | 0.5294 | 0.1471 | 0.2824 | 1.0000 | 0.4444 |
| long_context | 0.8333 | 0.1490 | 0.0630 | 0.0000 | 0.0000 |
| multi_hop | 0.5321 | 0.1090 | 0.1205 | 1.0000 | 0.5455 |
| rule_application | 0.6465 | 0.3687 | 0.2237 | 0.4286 | 0.3000 |
| temporal_factual | 0.7857 | 0.2429 | 0.1667 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5804 | 0.3810 | 0.3426 | 0.4167 | 0.3448 |
| hard | 0.4191 | 0.1155 | 0.2516 | 1.0000 | 0.9333 |
| medium | 0.6118 | 0.3465 | 0.3103 | 0.4615 | 0.3750 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5417 | 0.2702 | 0.3038 | 0.7812 | 0.7246 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
