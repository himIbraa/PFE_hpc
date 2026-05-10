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
| citation_f1 | 0.0764 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0646 |
| citation_recall | 0.1271 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4333 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.0802 |
| map_doc | 0.5104 |
| mean_latency_s | 0.0178 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2031 |
| mrr_doc | 0.5521 |
| ndcg_article | 0.2144 |
| ndcg_doc | 0.6327 |
| precision_article | 0.0250 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1271 |
| recall_doc | 0.8125 |
| rouge_l | 0.0511 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0755 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 1.0000 | 1.0000 | 0.3333 | 0.0000 | 0.0000 |
| layman | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.6667 | 0.5000 | 0.1111 | 0.0000 | 0.0000 |
| multi_hop | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.1250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.2917 | 0.1250 | 0.1667 | 0.0000 | 0.0000 |
| unanswerable | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6667 | 0.6667 | 0.2222 | 0.0000 | 0.0000 |
| hard | 0.5312 | 0.1562 | 0.0694 | 0.0000 | 0.0000 |
| medium | 0.5167 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5521 | 0.2031 | 0.0764 | 0.0000 | 0.0000 |
