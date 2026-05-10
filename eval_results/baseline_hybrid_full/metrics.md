# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7254 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0937 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0680 |
| citation_recall | 0.1954 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.4103 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.1492 |
| map_doc | 0.5981 |
| mean_latency_s | 0.0991 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2133 |
| mrr_doc | 0.6318 |
| ndcg_article | 0.2304 |
| ndcg_doc | 0.6702 |
| precision_article | 0.0340 |
| precision_doc | 0.0824 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1954 |
| recall_doc | 0.7459 |
| rouge_l | 0.0393 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0653 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6250 | 0.1250 | 0.0556 | 0.0000 | 0.0000 |
| exact_article | 0.6898 | 0.3853 | 0.1604 | 0.0000 | 0.0000 |
| layman | 0.6078 | 0.0588 | 0.0196 | 0.0000 | 0.0000 |
| long_context | 0.7353 | 0.1196 | 0.0714 | 0.0000 | 0.0000 |
| multi_hop | 0.6827 | 0.1250 | 0.0433 | 0.0000 | 0.0000 |
| rule_application | 0.7083 | 0.3040 | 0.1373 | 0.0000 | 0.0000 |
| temporal_factual | 0.6357 | 0.1714 | 0.0952 | 0.0000 | 0.0000 |
| unanswerable | 0.3546 | 0.0063 | 0.0083 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.6399 | 0.3345 | 0.1452 | 0.0000 | 0.0000 |
| hard | 0.5822 | 0.1304 | 0.0574 | 0.0000 | 0.0000 |
| medium | 0.6867 | 0.2337 | 0.1038 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6455 | 0.2243 | 0.0986 | 0.0000 | 0.0000 |
| fr | 0.3681 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
