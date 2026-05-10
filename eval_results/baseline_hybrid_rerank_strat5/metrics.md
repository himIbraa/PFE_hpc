# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.7500 |
| abstention_f1 | 0.0000 |
| abstention_precision | 0.0000 |
| abstention_recall | 0.0000 |
| answer_faithfulness | 0.0000 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.0936 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0700 |
| citation_recall | 0.1733 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3742 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0500 |
| map_article | 0.1161 |
| map_doc | 0.5354 |
| mean_latency_s | 0.8687 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2258 |
| mrr_doc | 0.5729 |
| ndcg_article | 0.2380 |
| ndcg_doc | 0.6229 |
| precision_article | 0.0350 |
| precision_doc | 0.0800 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.1733 |
| recall_doc | 0.7375 |
| rouge_l | 0.0434 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0729 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.3667 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.8667 | 0.6000 | 0.2214 | 0.0000 | 0.0000 |
| layman | 0.6500 | 0.2000 | 0.0667 | 0.0000 | 0.0000 |
| long_context | 0.7167 | 0.4500 | 0.1200 | 0.0000 | 0.0000 |
| multi_hop | 0.6667 | 0.0667 | 0.0500 | 0.0000 | 0.0000 |
| rule_application | 0.7000 | 0.4000 | 0.1571 | 0.0000 | 0.0000 |
| temporal_factual | 0.3167 | 0.0400 | 0.0667 | 0.0000 | 0.0000 |
| unanswerable | 0.3000 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7583 | 0.5000 | 0.1726 | 0.0000 | 0.0000 |
| hard | 0.5000 | 0.1517 | 0.0758 | 0.0000 | 0.0000 |
| medium | 0.5333 | 0.1000 | 0.0500 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5729 | 0.2258 | 0.0936 | 0.0000 | 0.0000 |
