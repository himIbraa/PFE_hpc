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
| citation_f1 | 0.1522 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.1125 |
| citation_recall | 0.3063 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4646 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.2373 |
| map_doc | 0.7656 |
| mean_latency_s | 0.7818 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2969 |
| mrr_doc | 0.7969 |
| ndcg_article | 0.3126 |
| ndcg_doc | 0.8164 |
| precision_article | 0.0562 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.3063 |
| recall_doc | 0.8438 |
| rouge_l | 0.0525 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0784 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.5179 | 0.0000 | 0.0000 |
| layman | 1.0000 | 0.5000 | 0.1667 | 0.0000 | 0.0000 |
| long_context | 1.0000 | 0.2500 | 0.2000 | 0.0000 | 0.0000 |
| multi_hop | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.6250 | 0.5000 | 0.1667 | 0.0000 | 0.0000 |
| unanswerable | 1.0000 | 0.1250 | 0.1667 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 1.0000 | 1.0000 | 0.4563 | 0.0000 | 0.0000 |
| hard | 0.7812 | 0.2188 | 0.1333 | 0.0000 | 0.0000 |
| medium | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.7969 | 0.2969 | 0.1522 | 0.0000 | 0.0000 |
