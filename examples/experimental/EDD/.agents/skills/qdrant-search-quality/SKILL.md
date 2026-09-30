---
name: qdrant-search-quality
description: "Diagnoses and improves Qdrant search relevance. Use when someone reports 'search results are bad', 'wrong results', 'low precision', 'low recall', 'irrelevant matches', 'missing expected results', or asks 'how to improve search quality?', 'which embedding model?', 'should I use hybrid search?', 'how to combine keyword and vector search / fusion / RRF / prefetch?', 'should I use reranking?', 'relevance feedback?', 'how to measure retrieval quality?', 'build a golden set', 'ground truth dataset', or 'how to score recall@k?'. Also use when search quality degrades after quantization, model change, or data growth."
allowed-tools:
  - Read
  - Grep
  - Glob
---

# Qdrant Search Quality

Route first, then answer. Match the user's symptom in the table, `Read` that file, and answer from it.
Do not answer from this page alone: it contains routing only, not the guidance. If two rows match, read both.

| The user says | Read |
|---|---|
| Search results are bad or irrelevant, wrong results, missing expected matches | `diagnosis/SKILL.md` |
| Low recall, expected results are missing | `diagnosis/SKILL.md` |
| Low precision, too many wrong matches | `diagnosis/SKILL.md` |
| Which embedding model to use, quality dropped after quantization, model change, or data growth | `diagnosis/SKILL.md` |
| Not sure if the model, the data, or Qdrant is at fault | `diagnosis/SKILL.md` |
| Want to measure recall, build a golden set, ground truth dataset, recall@k | `diagnosis/SKILL.md` |
| Need to combine keyword and semantic search, hybrid search, sparse + dense, fusion / RRF, prefetch | `search-strategies/hybrid-search/SKILL.md` |
| Should I rerank, results too similar, need diversity, MMR, recommendation/discovery API | `search-strategies/SKILL.md` |
| Improving results with relevance feedback or user clicks, cheaper alternative to reranking | `search-strategies/relevance-feedback/SKILL.md` |

Most quality issues come from the embedding model or the data, not from Qdrant's configuration — splitting chunks mid-sentence alone can drop quality 30-40%.
Rule that out with exact search before tuning any Qdrant parameter:
[Search API](https://skills.qdrant.tech/md/documentation/search/search/?s=search-api)
