---
name: qdrant-hybrid-search-combining
description: "Fusing scores from multiple searches into a single ranked result (RRF, DBSF, custom fusion). Use when someone asks 'RRF or DBSF?', 'how to combine sparse and dense', 'how to combine scores from multiple searches?', 'custom fusion', or 'fusion is not producing good results'"
---

# Combining Prefetch Results

The outer query fuses ranked candidate lists from all parallel prefetches into one ranked list of results. Fusion methods differ in whether they use rank, score or directly vector representations of candidates (their similarity to the outer query) and whether final score incorporates payload metadata. All methods support flat (one fusion step) and nested (multi-stage) prefetch structures.

## Scores Are Not Comparable Across Prefetches & You Want Some Easy Baseline

Use when: searches produce scores on different scales, like BM25 and cosine on dense embeddings.

### RRF
- **[RRF](https://skills.qdrant.tech/md/documentation/search/hybrid-queries/?s=reciprocal-rank-fusion-rrf)** (Reciprocal Rank Fusion) — rank-based, ignores scores magnitude, a decent default to start with.
- Tune `k` to [control rank sensitivity in RRF fusion](https://skills.qdrant.tech/md/documentation/search/hybrid-queries/?s=setting-rrf-constant-k).
- Add per-prefetch **weights** when one search should dominate, using [Weighted RRF](https://skills.qdrant.tech/md/documentation/search/hybrid-queries/?s=weighted-rrf). Weights should be customized per collection and retrievers' score distributions!

### DBSF
- **[DBSF](https://skills.qdrant.tech/md/documentation/search/hybrid-queries/?s=distribution-based-score-fusion-dbsf)** (Distribution-Based Score Fusion) — normalizes score distributions per prefetch before fusing them, for that, instead of min-max, uses mean +- 3 deviations on prefetched list of scores. Avoid relying on resulting absolute scores, as scores in DBSF are normalized per prefetch (aka per a retrieved list of search results), and might be uncomparable across queries.

## Need Custom Fusion

Use when: recency, popularity or other payload values should affect the merged ranking alongside candidate scores or you need a custom fusion.

**[With formula query](https://skills.qdrant.tech/md/documentation/search/search-relevance/?s=score-boosting)**, access `score` of each prefetch and, if desired, payload field values.

If you want to implement custom fusion on `score` of each prefetch:
- Use decay or any other available expressions for normalizing score distributions before fusing them. 
- Parameters of these expressions should be based on the collection & retriever score distributions (for example, adjusting these parameters on a subsample of real queries). 
- Formula query is unable to provide ranks for custom fusions 

When using `FormulaQuery` over multiple prefetches (e.g. per-representation weighting):
- `$score[i]` indexes prefetches in declaration order. Reordering the `prefetch=` list silently shifts which weight applies to which retriever.
- Provide `defaults` for every `$score[i]` so the formula still evaluates for candidates that surfaced from only a subset of prefetches.
- Start with RRF when scores are on incomparable scales (e.g. BM25 + cosine). Reach for `FormulaQuery` only when explicit per-representation weighting or payload-driven boosts are required, and normalize each `$score[i]` (decay or min-max on a sampled distribution) before combining linearly.

## Need Good Ranking of Fused Candidates and Ready To Spend More Resources

Use when: you want to use similarity between query and candidates' vector representations as the prefetches combiner and simultaneously ranker. 
More resource heavy than score/rank based fusions, but might be necessary due to use case requirements or need in a high top-K precision of results (when parallel prefetches have overall a good recall of retrieved candidates).

You can use any type of vector as an outer query over the prefetches, to perform the fusion on the server-side in one QueryAPI request: sparse, dense, multivector. For that, same type of vector representations for documents need to be stored as named vectors per point.

Instead of using client-side fusion through cross-encoders, a popular option is **Late interaction models-based fusion**, through reranking on multivectors (e.g. ColBERT for text, ColPali and ColQwen for images).
- Most precise but highest compute/resource usage.
- Configure multivectors used for fusion through reranking with HNSW disabled like in [Hybrid Search with Reranking tutorial](https://skills.qdrant.tech/md/documentation/tutorials-basics/reranking-hybrid-search/).

## What NOT to Do

- Use linear weighted fusion on incomparable score ranges. [Why not](https://skills.qdrant.tech/md/articles/hybrid-search/?s=fusion-merges-two-rankings-into-one).
- Use "vibe" defined weights in weighted RRF. Weights should be fine-tuned per dataset and retrieval pipelines.
- Pick any fusion type without comparative experiments.
- Use late interaction multivectors for fusion without evaluating cheaper analogues, for example, MUVERA. More in [multi-vector Qdrant search course](https://skills.qdrant.tech/md/course/multi-vector-search/)
