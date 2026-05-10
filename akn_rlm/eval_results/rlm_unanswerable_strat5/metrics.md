# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.6000 |
| abstention_f1 | 0.4667 |
| abstention_precision | 0.3500 |
| abstention_recall | 0.7000 |
| answer_faithfulness | 0.5000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.1689 |
| citation_groundedness | 0.5000 |
| citation_precision | 0.0700 |
| citation_recall | 0.1725 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3692 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1151 |
| map_doc | 0.3688 |
| mean_latency_s | 0.3542 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.1633 |
| mrr_doc | 0.3833 |
| ndcg_article | 0.1762 |
| ndcg_doc | 0.3936 |
| precision_article | 0.0350 |
| precision_doc | 0.0475 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1725 |
| recall_doc | 0.4125 |
| rouge_l | 0.0285 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0429 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.6000 | 0.6000 | 0.3214 | 0.5000 | 0.6667 |
| layman | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| long_context | 0.6000 | 0.1000 | 0.0800 | 0.0000 | 0.0000 |
| multi_hop | 0.7000 | 0.2000 | 0.0500 | 0.0000 | 0.0000 |
| rule_application | 0.2000 | 0.1000 | 0.1000 | 0.0000 | 0.0000 |
| temporal_factual | 0.3667 | 0.3067 | 0.2000 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.4000 | 0.3000 | 0.1607 | 0.5000 | 0.4444 |
| hard | 0.4167 | 0.1517 | 0.2325 | 1.0000 | 0.7143 |
| medium | 0.3000 | 0.0500 | 0.0500 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.3833 | 0.1633 | 0.1689 | 0.7000 | 0.4667 |
