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
| citation_f1 | 0.0804 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0562 |
| citation_recall | 0.1771 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3854 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0625 |
| map_article | 0.1536 |
| map_doc | 0.6042 |
| mean_latency_s | 0.7690 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2031 |
| mrr_doc | 0.6510 |
| ndcg_article | 0.2144 |
| ndcg_doc | 0.7077 |
| precision_article | 0.0250 |
| precision_doc | 0.0875 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1771 |
| recall_doc | 0.8125 |
| rouge_l | 0.0535 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0833 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.6667 | 0.6250 | 0.3095 | 0.0000 | 0.0000 |
| layman | 0.7500 | 0.5000 | 0.1667 | 0.0000 | 0.0000 |
| long_context | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| multi_hop | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| rule_application | 0.1667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| temporal_factual | 0.6250 | 0.5000 | 0.1667 | 0.0000 | 0.0000 |
| unanswerable | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7778 | 0.7500 | 0.3175 | 0.0000 | 0.0000 |
| hard | 0.7812 | 0.1250 | 0.0417 | 0.0000 | 0.0000 |
| medium | 0.3667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6510 | 0.2031 | 0.0804 | 0.0000 | 0.0000 |
