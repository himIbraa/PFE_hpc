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
| citation_f1 | 0.0841 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0658 |
| citation_recall | 0.1433 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3792 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0250 |
| map_article | 0.0964 |
| map_doc | 0.5417 |
| mean_latency_s | 0.0170 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2000 |
| mrr_doc | 0.5750 |
| ndcg_article | 0.2083 |
| ndcg_doc | 0.6319 |
| precision_article | 0.0300 |
| precision_doc | 0.0800 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1433 |
| recall_doc | 0.7500 |
| rouge_l | 0.0422 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0704 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4667 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |
| exact_article | 0.7000 | 0.6000 | 0.2476 | 0.0000 | 0.0000 |
| layman | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.7667 | 0.3000 | 0.0844 | 0.0000 | 0.0000 |
| multi_hop | 0.5333 | 0.2000 | 0.0500 | 0.0000 | 0.0000 |
| rule_application | 0.6500 | 0.4000 | 0.1571 | 0.0000 | 0.0000 |
| temporal_factual | 0.4833 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |
| unanswerable | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6500 | 0.4000 | 0.1524 | 0.0000 | 0.0000 |
| hard | 0.5458 | 0.1375 | 0.0503 | 0.0000 | 0.0000 |
| medium | 0.5583 | 0.1250 | 0.0833 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5750 | 0.2000 | 0.0841 | 0.0000 | 0.0000 |
