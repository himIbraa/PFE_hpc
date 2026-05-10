# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8033 |
| abstention_f1 | 0.6842 |
| abstention_precision | 0.6118 |
| abstention_recall | 0.7761 |
| answer_faithfulness | 0.3484 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2996 |
| citation_groundedness | 0.5255 |
| citation_precision | 0.1893 |
| citation_recall | 0.2127 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6066 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1880 |
| map_doc | 0.5096 |
| mean_latency_s | 4.0848 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2645 |
| mrr_doc | 0.5389 |
| ndcg_article | 0.2766 |
| ndcg_doc | 0.5493 |
| precision_article | 0.0352 |
| precision_doc | 0.0607 |
| reasoning_chain_score | 0.0055 |
| recall_article | 0.2127 |
| recall_doc | 0.5458 |
| rouge_l | 0.0896 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1154 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4583 | 0.1250 | 0.1528 | 0.0000 | 0.0000 |
| exact_article | 0.6017 | 0.4718 | 0.4128 | 0.4167 | 0.3333 |
| layman | 0.5294 | 0.1471 | 0.2824 | 1.0000 | 0.4444 |
| long_context | 0.8333 | 0.1490 | 0.0862 | 0.0000 | 0.0000 |
| multi_hop | 0.6346 | 0.1859 | 0.1493 | 0.6667 | 0.4444 |
| rule_application | 0.6869 | 0.3611 | 0.2189 | 0.4286 | 0.2857 |
| temporal_factual | 0.7857 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5893 | 0.3899 | 0.3491 | 0.4167 | 0.3448 |
| hard | 0.4515 | 0.1330 | 0.2643 | 0.9762 | 0.9111 |
| medium | 0.6118 | 0.3412 | 0.3099 | 0.4615 | 0.3636 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5582 | 0.2782 | 0.3108 | 0.7656 | 0.7050 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
