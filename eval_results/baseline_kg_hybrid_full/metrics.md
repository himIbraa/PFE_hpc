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
| citation_f1 | 0.0833 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0598 |
| citation_recall | 0.1754 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3696 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0123 |
| map_article | 0.1171 |
| map_doc | 0.4925 |
| mean_latency_s | 14.7580 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1800 |
| mrr_doc | 0.5296 |
| ndcg_article | 0.1981 |
| ndcg_doc | 0.5652 |
| precision_article | 0.0299 |
| precision_doc | 0.0701 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1754 |
| recall_doc | 0.6223 |
| rouge_l | 0.0358 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0609 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.3889 | 0.1069 | 0.1111 | 0.0000 | 0.0000 |
| exact_article | 0.6441 | 0.3198 | 0.1392 | 0.0000 | 0.0000 |
| layman | 0.3382 | 0.0147 | 0.0196 | 0.0000 | 0.0000 |
| long_context | 0.7059 | 0.0902 | 0.0382 | 0.0000 | 0.0000 |
| multi_hop | 0.6859 | 0.0897 | 0.0337 | 0.0000 | 0.0000 |
| rule_application | 0.6242 | 0.2801 | 0.1151 | 0.0000 | 0.0000 |
| temporal_factual | 0.1429 | 0.0952 | 0.0952 | 0.0000 | 0.0000 |
| unanswerable | 0.2196 | 0.0125 | 0.0167 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5312 | 0.2574 | 0.1110 | 0.0000 | 0.0000 |
| hard | 0.4914 | 0.0945 | 0.0478 | 0.0000 | 0.0000 |
| medium | 0.5749 | 0.2325 | 0.1083 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5423 | 0.1893 | 0.0877 | 0.0000 | 0.0000 |
| fr | 0.2847 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
