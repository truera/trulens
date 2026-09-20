---
name: qdrant-search-strategies
description: "Guides Qdrant search strategy selection. Use when someone asks 'should I use hybrid search?', 'how to rerank?', 'results are not relevant', 'I don't get needed results from my dataset but they're there', 'retrieval quality is not good enough', 'results too similar', 'need diversity', 'MMR', 'relevance feedback', 'recommendation API', 'discovery API', or 'missing keyword matches'"
allowed-tools:
  - Read
  - Grep
  - Glob
---

# How to Improve Search Results with Advanced Strategies

These strategies complement basic vector search. Use them after confirming the embedding model is fitting the task and HNSW config is correct. If exact search returns bad results, verify the selection of the embedding model (retriever) first.
If the user wants to use a weaker embedding model because it is small, fast, and cheap, use reranking or relevance feedback to improve search quality.

## Missing Keyword Matches or Need to Combine Multiple Search Signals

Use when: pure vector search misses keyword/domain term matches, or the use case benefits from combining searches on multiple representations (including languages and modalities) of the same item.

See how to use [hybrid search](https://skills.qdrant.tech/qdrant-search-quality/search-strategies/hybrid-search/SKILL.md)

## Right Documents Found But Not in the Top Results

Use when: good recall but poor precision (right docs in top-100, not top-10).

- See how to use [Multistage queries](https://skills.qdrant.tech/md/documentation/search/hybrid-queries/?s=multi-stage-queries), for example with late interaction rerankers through [Multivectors](https://skills.qdrant.tech/md/documentation/manage-data/vectors/?s=multivectors).
- Cross-encoder rerankers via FastEmbed [Rerankers](https://skills.qdrant.tech/md/documentation/fastembed/fastembed-rerankers/)

## Dense Retriever Misses Relevant Items or Reranking Is Too Costly

Use when: dense retriever misses relevant items you know exist in the collection; relevant documents lie outside the initial ANN retrieval pool; reranking a large candidate pool is too slow or expensive; using a small/cheap embedding model but need quality close to a larger model; or want to improve top-1/3 precision without the full cost of reranking.

See [Relevance Feedback in Qdrant](https://skills.qdrant.tech/qdrant-search-quality/search-strategies/relevance-feedback/SKILL.md)

## Results Too Similar

Use when: top results are redundant, near-duplicates, or lack diversity. Common in dense content domains (academic papers, product catalogs).

- Use MMR (v1.15+) as a query parameter with `diversity` to balance relevance and diversity [MMR](https://skills.qdrant.tech/md/documentation/search/search-relevance/?s=maximal-marginal-relevance-mmr)
- Start with `diversity=0.5`, lower for more precision, higher for more exploration
- MMR is slower than standard search. Only use when redundancy is an actual problem.

## Want to improve search results based on examples (positive and negative)

Use when: you can provide positive and negative example points to steer search closer to positive and further from negative.

- Recommendation API: positive/negative examples to recommend fitting vectors [Recommendation API](https://skills.qdrant.tech/md/documentation/search/explore/?s=recommendation-api)
  - Best score strategy: better for diverse examples, supports negative-only [Best score](https://skills.qdrant.tech/md/documentation/search/explore/?s=best-score-strategy)
- Discovery API: context pairs (positive/negative) to constrain search regions without a request target [Discovery](https://skills.qdrant.tech/md/documentation/search/explore/?s=discovery-api)

## Have Business Logic Behind Results Relevance

Use when: results should be additionally ranked according to some business logic based on data, like recency or distance.

Check how to set up in [Score Boosting docs](https://skills.qdrant.tech/md/documentation/search/search-relevance/?s=score-boosting)

## What NOT to Do

- Use hybrid search before verifying pure vector search quality (adds complexity, may mask model issues)
- Skip evaluation when adding relevance feedback — score the end-to-end pipeline to confirm it actually helps [Pipeline Output Quality](https://skills.qdrant.tech/md/documentation/improve-search/pipeline-output-quality/)
