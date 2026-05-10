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
| citation_f1 | 0.0633 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0447 |
| citation_recall | 0.1424 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3622 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0082 |
| map_article | 0.0961 |
| map_doc | 0.4931 |
| mean_latency_s | 0.0957 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1418 |
| mrr_doc | 0.5300 |
| ndcg_article | 0.1576 |
| ndcg_doc | 0.5748 |
| precision_article | 0.0221 |
| precision_doc | 0.0746 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1424 |
| recall_doc | 0.6653 |
| rouge_l | 0.0339 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0546 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5694 | 0.1250 | 0.1071 | 0.0000 | 0.0000 |
| exact_article | 0.5418 | 0.2746 | 0.1178 | 0.0000 | 0.0000 |
| layman | 0.4412 | 0.0588 | 0.0196 | 0.0000 | 0.0000 |
| long_context | 0.6716 | 0.0196 | 0.0107 | 0.0000 | 0.0000 |
| multi_hop | 0.7096 | 0.1423 | 0.0476 | 0.0000 | 0.0000 |
| rule_application | 0.5725 | 0.1571 | 0.0725 | 0.0000 | 0.0000 |
| temporal_factual | 0.4643 | 0.2143 | 0.0952 | 0.0000 | 0.0000 |
| unanswerable | 0.3029 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.4765 | 0.2298 | 0.1012 | 0.0000 | 0.0000 |
| hard | 0.5217 | 0.1006 | 0.0438 | 0.0000 | 0.0000 |
| medium | 0.5753 | 0.1337 | 0.0619 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5229 | 0.1491 | 0.0666 | 0.0000 | 0.0000 |
| fr | 0.6667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
