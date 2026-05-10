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
| citation_f1 | 0.1052 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0754 |
| citation_recall | 0.2196 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4387 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0287 |
| map_article | 0.1641 |
| map_doc | 0.5874 |
| mean_latency_s | 0.3975 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2417 |
| mrr_doc | 0.6214 |
| ndcg_article | 0.2584 |
| ndcg_doc | 0.6599 |
| precision_article | 0.0377 |
| precision_doc | 0.0828 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.2196 |
| recall_doc | 0.7350 |
| rouge_l | 0.0386 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0651 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4861 | 0.1250 | 0.0516 | 0.0000 | 0.0000 |
| exact_article | 0.6737 | 0.4489 | 0.1834 | 0.0000 | 0.0000 |
| layman | 0.4706 | 0.0588 | 0.0196 | 0.0000 | 0.0000 |
| long_context | 0.6716 | 0.2324 | 0.0738 | 0.0000 | 0.0000 |
| multi_hop | 0.7321 | 0.1122 | 0.0542 | 0.0000 | 0.0000 |
| rule_application | 0.7563 | 0.3283 | 0.1551 | 0.0000 | 0.0000 |
| temporal_factual | 0.5119 | 0.1714 | 0.0952 | 0.0000 | 0.0000 |
| unanswerable | 0.3521 | 0.0063 | 0.0083 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6265 | 0.3955 | 0.1516 | 0.0000 | 0.0000 |
| hard | 0.5574 | 0.1277 | 0.0583 | 0.0000 | 0.0000 |
| medium | 0.6955 | 0.2784 | 0.1315 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6219 | 0.2499 | 0.1092 | 0.0000 | 0.0000 |
| fr | 0.6111 | 0.0833 | 0.0278 | 0.0000 | 0.0000 |
