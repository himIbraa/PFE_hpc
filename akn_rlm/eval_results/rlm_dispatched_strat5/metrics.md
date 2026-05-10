# Evaluation Metrics

**Total questions:** 40

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8250 |
| abstention_f1 | 0.6316 |
| abstention_precision | 0.6667 |
| abstention_recall | 0.6000 |
| answer_faithfulness | 0.2250 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.2679 |
| citation_groundedness | 0.5750 |
| citation_precision | 0.2038 |
| citation_recall | 0.2546 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6042 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0000 |
| map_article | 0.1889 |
| map_doc | 0.5938 |
| mean_latency_s | 5.1046 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2821 |
| mrr_doc | 0.6125 |
| ndcg_article | 0.3031 |
| ndcg_doc | 0.6289 |
| precision_article | 0.0475 |
| precision_doc | 0.0700 |
| reasoning_chain_score | 0.0129 |
| recall_article | 0.2546 |
| recall_doc | 0.6500 |
| rouge_l | 0.0988 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1336 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.6000 | 0.0667 | 0.0667 | 0.0000 | 0.0000 |
| exact_article | 0.8000 | 0.6000 | 0.4133 | 0.2500 | 0.4000 |
| layman | 0.6000 | 0.4000 | 0.3333 | 0.0000 | 0.0000 |
| long_context | 0.9000 | 0.1500 | 0.0800 | 0.0000 | 0.0000 |
| multi_hop | 0.7000 | 0.3000 | 0.1500 | 0.0000 | 0.0000 |
| rule_application | 0.6000 | 0.4000 | 0.2667 | 0.0000 | 0.0000 |
| temporal_factual | 0.7000 | 0.3400 | 0.2333 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.7000 | 0.5000 | 0.3733 | 0.2500 | 0.2857 |
| hard | 0.5750 | 0.1975 | 0.2658 | 1.0000 | 0.9091 |
| medium | 0.6000 | 0.2333 | 0.1667 | 0.0000 | 0.0000 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.6125 | 0.2821 | 0.2679 | 0.6000 | 0.6316 |
