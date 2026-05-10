# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7049 |
| abstention_f1 | 0.0769 |
| abstention_precision | 0.2727 |
| abstention_recall | 0.0448 |
| answer_faithfulness | 0.0451 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0223 |
| citation_groundedness | 0.9549 |
| citation_precision | 0.0139 |
| citation_recall | 0.0343 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.0954 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.0196 |
| map_doc | 0.1110 |
| mean_latency_s | 14.8069 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.0348 |
| mrr_doc | 0.1185 |
| ndcg_article | 0.0419 |
| ndcg_doc | 0.1313 |
| precision_article | 0.0070 |
| precision_doc | 0.0172 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.0343 |
| recall_doc | 0.1578 |
| rouge_l | 0.0275 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0479 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.2083 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.1003 | 0.0579 | 0.0307 | 0.0000 | 0.0000 |
| layman | 0.1176 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.0294 | 0.0294 | 0.0118 | 0.0000 | 0.0000 |
| multi_hop | 0.1090 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.1818 | 0.0616 | 0.0318 | 0.0000 | 0.0000 |
| temporal_factual | 0.0714 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| unanswerable | 0.0667 | 0.0125 | 0.0333 | 0.0750 | 0.1395 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.0863 | 0.0476 | 0.0213 | 0.0000 | 0.0000 |
| hard | 0.1052 | 0.0265 | 0.0268 | 0.0714 | 0.1333 |
| medium | 0.1559 | 0.0363 | 0.0175 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.1246 | 0.0366 | 0.0191 | 0.0000 | 0.0000 |
| fr | 0.0000 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
