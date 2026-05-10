# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8238 |
| abstention_f1 | 0.7034 |
| abstention_precision | 0.6538 |
| abstention_recall | 0.7612 |
| answer_faithfulness | 0.3197 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2933 |
| citation_groundedness | 0.5327 |
| citation_precision | 0.1878 |
| citation_recall | 0.2104 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6079 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1866 |
| map_doc | 0.5109 |
| mean_latency_s | 4.1799 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2670 |
| mrr_doc | 0.5444 |
| ndcg_article | 0.2778 |
| ndcg_doc | 0.5554 |
| precision_article | 0.0357 |
| precision_doc | 0.0607 |
| reasoning_chain_score | 0.0057 |
| recall_article | 0.2104 |
| recall_doc | 0.5478 |
| rouge_l | 0.0903 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1170 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4583 | 0.0833 | 0.0972 | 0.0000 | 0.0000 |
| exact_article | 0.6186 | 0.4703 | 0.4139 | 0.4167 | 0.3571 |
| layman | 0.5294 | 0.1471 | 0.2689 | 1.0000 | 0.4444 |
| long_context | 0.8333 | 0.1490 | 0.0862 | 0.0000 | 0.0000 |
| multi_hop | 0.6667 | 0.1744 | 0.1198 | 0.3333 | 0.2857 |
| rule_application | 0.6793 | 0.3838 | 0.2197 | 0.4286 | 0.3333 |
| temporal_factual | 0.7857 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6071 | 0.3884 | 0.3515 | 0.4167 | 0.3571 |
| hard | 0.4644 | 0.1398 | 0.2590 | 0.9524 | 0.9412 |
| medium | 0.6000 | 0.3412 | 0.2966 | 0.4615 | 0.3750 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5639 | 0.2808 | 0.3042 | 0.7500 | 0.7273 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
