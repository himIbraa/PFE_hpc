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
| citation_f1 | 0.1136 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0800 |
| citation_recall | 0.2433 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3183 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1236 |
| map_doc | 0.4625 |
| mean_latency_s | 15.4446 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2037 |
| mrr_doc | 0.4896 |
| ndcg_article | 0.2371 |
| ndcg_doc | 0.5429 |
| precision_article | 0.0400 |
| precision_doc | 0.0700 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.2433 |
| recall_doc | 0.6500 |
| rouge_l | 0.0412 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0662 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4333 | 0.0900 | 0.1333 | 0.0000 | 0.0000 |
| exact_article | 0.8000 | 0.6000 | 0.2143 | 0.0000 | 0.0000 |
| layman | 0.4500 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |
| long_context | 0.6000 | 0.1067 | 0.0800 | 0.0000 | 0.0000 |
| multi_hop | 0.5333 | 0.2000 | 0.0500 | 0.0000 | 0.0000 |
| rule_application | 0.4667 | 0.4000 | 0.1643 | 0.0000 | 0.0000 |
| temporal_factual | 0.1333 | 0.1333 | 0.1333 | 0.0000 | 0.0000 |
| unanswerable | 0.5000 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6750 | 0.4250 | 0.1976 | 0.0000 | 0.0000 |
| hard | 0.4417 | 0.1225 | 0.0825 | 0.0000 | 0.0000 |
| medium | 0.4000 | 0.1450 | 0.0917 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.4896 | 0.2037 | 0.1136 | 0.0000 | 0.0000 |
