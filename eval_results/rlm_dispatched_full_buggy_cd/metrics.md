# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7992 |
| abstention_f1 | 0.6879 |
| abstention_precision | 0.6000 |
| abstention_recall | 0.8060 |
| answer_faithfulness | 0.3689 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2892 |
| citation_groundedness | 0.4919 |
| citation_precision | 0.1784 |
| citation_recall | 0.2144 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.5753 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1802 |
| map_doc | 0.4863 |
| mean_latency_s | 3.7732 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2533 |
| mrr_doc | 0.5130 |
| ndcg_article | 0.2674 |
| ndcg_doc | 0.5247 |
| precision_article | 0.0348 |
| precision_doc | 0.0574 |
| reasoning_chain_score | 0.0050 |
| recall_article | 0.2144 |
| recall_doc | 0.5280 |
| rouge_l | 0.0886 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1130 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5417 | 0.1319 | 0.1071 | 0.0000 | 0.0000 |
| exact_article | 0.5763 | 0.4788 | 0.4111 | 0.5000 | 0.3871 |
| layman | 0.5294 | 0.1471 | 0.2745 | 1.0000 | 0.4444 |
| long_context | 0.8333 | 0.1490 | 0.0630 | 0.0000 | 0.0000 |
| multi_hop | 0.5064 | 0.0962 | 0.1250 | 1.0000 | 0.4615 |
| rule_application | 0.6490 | 0.3447 | 0.2103 | 0.4286 | 0.2857 |
| temporal_factual | 0.7857 | 0.2429 | 0.1667 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5804 | 0.3973 | 0.3456 | 0.5000 | 0.4138 |
| hard | 0.4110 | 0.1163 | 0.2540 | 1.0000 | 0.8936 |
| medium | 0.5922 | 0.3245 | 0.2948 | 0.4615 | 0.3529 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5309 | 0.2665 | 0.2999 | 0.7969 | 0.7083 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
