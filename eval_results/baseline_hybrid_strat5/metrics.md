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
| citation_f1 | 0.1144 |
| citation_groundedness | 1.0000 |
| citation_precision | 0.0850 |
| citation_recall | 0.2150 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.3858 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1565 |
| map_doc | 0.6446 |
| mean_latency_s | 0.6086 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2162 |
| mrr_doc | 0.6696 |
| ndcg_article | 0.2367 |
| ndcg_doc | 0.7031 |
| precision_article | 0.0425 |
| precision_doc | 0.0825 |
| reasoning_chain_score | 0.0000 |
| recall_article | 0.2150 |
| recall_doc | 0.7750 |
| rouge_l | 0.0461 |
| sacrebleu | 0.0000 |
| token_f1 | 0.0714 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| exact_article | 0.6667 | 0.6000 | 0.2643 | 0.0000 | 0.0000 |
| layman | 0.8000 | 0.2000 | 0.0667 | 0.0000 | 0.0000 |
| long_context | 0.8000 | 0.1400 | 0.1200 | 0.0000 | 0.0000 |
| multi_hop | 0.5000 | 0.2000 | 0.0500 | 0.0000 | 0.0000 |
| rule_application | 0.8000 | 0.3000 | 0.2143 | 0.0000 | 0.0000 |
| temporal_factual | 0.5900 | 0.2400 | 0.1333 | 0.0000 | 0.0000 |
| unanswerable | 0.6000 | 0.0500 | 0.0667 | 0.0000 | 0.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7333 | 0.5000 | 0.2226 | 0.0000 | 0.0000 |
| hard | 0.6225 | 0.1575 | 0.0925 | 0.0000 | 0.0000 |
| medium | 0.7000 | 0.0500 | 0.0500 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6696 | 0.2162 | 0.1144 | 0.0000 | 0.0000 |
