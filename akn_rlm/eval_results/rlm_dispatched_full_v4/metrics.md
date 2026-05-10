# Evaluation Metrics

**Total questions:** 244

## Overall

| Metric | Score |
|--------|------:|
| abstention_acc | 0.8238 |
| abstention_f1 | 0.7075 |
| abstention_precision | 0.6500 |
| abstention_recall | 0.7761 |
| answer_faithfulness | 0.3279 |
| bertscore_f1 | 0.0000 |
| citation_f1 | 0.3011 |
| citation_groundedness | 0.5475 |
| citation_precision | 0.1941 |
| citation_recall | 0.2155 |
| corrective_retry_rate | 0.0000 |
| doc_citation_f1 | 0.6122 |
| exact_match | 0.0000 |
| hcr | 0.0000 |
| jir | 0.0041 |
| map_article | 0.1856 |
| map_doc | 0.5085 |
| mean_latency_s | 4.5086 |
| mean_tokens_per_query | 0.0000 |
| mrr_article | 0.2686 |
| mrr_doc | 0.5567 |
| ndcg_article | 0.2804 |
| ndcg_doc | 0.5654 |
| precision_article | 0.0361 |
| precision_doc | 0.0594 |
| reasoning_chain_score | 0.0057 |
| recall_article | 0.2155 |
| recall_doc | 0.5389 |
| rouge_l | 0.0917 |
| sacrebleu | 0.0000 |
| token_f1 | 0.1170 |

## By Query Type

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| conceptual_definitional | 0.5417 | 0.1625 | 0.1349 | 0.0000 | 0.0000 |
| exact_article | 0.5763 | 0.4548 | 0.4164 | 0.4167 | 0.3333 |
| layman | 0.5294 | 0.1471 | 0.2745 | 1.0000 | 0.4444 |
| long_context | 0.8529 | 0.2451 | 0.0966 | 0.0000 | 0.0000 |
| multi_hop | 0.7692 | 0.1090 | 0.1216 | 0.6667 | 0.5714 |
| rule_application | 0.7020 | 0.3902 | 0.2346 | 0.4286 | 0.3333 |
| temporal_factual | 0.7857 | 0.2143 | 0.1905 | 0.0000 | 0.0000 |
| unanswerable | 0.0000 | 0.0000 | 0.5250 | 1.0000 | 1.0000 |

## By Difficulty

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| easy | 0.5804 | 0.3720 | 0.3435 | 0.4167 | 0.3448 |
| hard | 0.4984 | 0.1383 | 0.2653 | 0.9762 | 0.9647 |
| medium | 0.6118 | 0.3582 | 0.3166 | 0.4615 | 0.3636 |

## By Language

| Stratum | mrr_doc | mrr_article | citation_f1 | abstention_recall | abstention_f1 |
|---------|------:|------:|------:|------:|------:|
| ar | 0.5769 | 0.2825 | 0.3124 | 0.7656 | 0.7313 |
| fr | 0.1667 | 0.0000 | 0.0833 | 1.0000 | 0.4615 |
