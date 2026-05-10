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
| citation_f1 | 0.0523 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0375 |
| citation_recall | 0.1104 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.2992 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0250 |
| map_article | 0.0885 |
| map_doc | 0.4667 |
| mean_latency_s | 0.3338 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1437 |
| mrr_doc | 0.5104 |
| ndcg_article | 0.1515 |
| ndcg_doc | 0.5585 |
| precision_article | 0.0175 |
| precision_doc | 0.0700 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1104 |
| recall_doc | 0.6500 |
| rouge_l | 0.0398 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0614 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.4167 | 0.2500 | 0.1238 | 0.0000 | 0.0000 |
| layman | 0.5000 | 0.2000 | 0.0667 | 0.0000 | 0.0000 |
| long_context | 0.6500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi_hop | 0.9000 | 0.4000 | 0.0944 | 0.0000 | 0.0000 |
| rule_application | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.3500 | 0.3000 | 0.1333 | 0.0000 | 0.0000 |
| unanswerable | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.4083 | 0.2250 | 0.0952 | 0.0000 | 0.0000 |
| hard | 0.6250 | 0.1750 | 0.0569 | 0.0000 | 0.0000 |
| medium | 0.3833 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5104 | 0.1437 | 0.0523 | 0.0000 | 0.0000 |
