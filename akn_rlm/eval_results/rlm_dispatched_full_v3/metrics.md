# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8115 |
| abstention_f1 | 0.6933 |
| abstention_precision | 0.6265 |
| abstention_recall | 0.7761 |
| answer_faithfulness | 0.3402 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2980 |
| citation_groundedness | 0.5460 |
| citation_precision | 0.2003 |
| citation_recall | 0.2100 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6053 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1860 |
| map_doc | 0.4969 |
| mean_latency_s | 4.7614 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2609 |
| mrr_doc | 0.5273 |
| ndcg_article | 0.2717 |
| ndcg_doc | 0.5394 |
| precision_article | 0.0336 |
| precision_doc | 0.0598 |
| reasoning_chain_score | 0.0054 |
| recall_article | 0.2100 |
| recall_doc | 0.5410 |
| rouge_l | 0.0886 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1143 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.1486 | 0.1349 | 0.0000 | 0.0000 |
| exact_article | 0.5847 | 0.4633 | 0.4299 | 0.5000 | 0.4000 |
| layman | 0.5294 | 0.1765 | 0.2235 | 0.5000 | 0.2500 |
| long_context | 0.8137 | 0.2225 | 0.0982 | 0.0000 | 0.0000 |
| multi_hop | 0.6731 | 0.0962 | 0.1181 | 0.6667 | 0.5714 |
| rule_application | 0.6414 | 0.3598 | 0.2250 | 0.4286 | 0.2727 |
| temporal_factual | 0.7857 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5714 | 0.3810 | 0.3458 | 0.4167 | 0.3448 |
| hard | 0.4434 | 0.1265 | 0.2627 | 0.9762 | 0.9318 |
| medium | 0.6000 | 0.3445 | 0.3093 | 0.4615 | 0.3636 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5460 | 0.2744 | 0.3091 | 0.7656 | 0.7153 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
