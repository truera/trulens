---
name: qdrant-relevance-feedback
description: "Expanding the candidate pool via relevance feedback, as an alternative to reranking when a dense retriever is too weak. Use when someone asks about 'Qdrant's Relevance Feedback API', 'improving dense search relevance/recall', 'how to discover/get more relevant results from vector search', 'cheaper/better alternative to reranking', 'using a more heavy/big embedding model for dense search but can't afford it', 'finding more relevant documents beyond the initial search pool', or 'feedback loops'. Also trigger when the user has a search quality problem due to a dense retriever being weak and is considering reranking as a solution — this API may be a better fit"
---

Reranking reorders documents that have already been retrieved. Qdrant's Relevance Feedback (RF) instead modifies the vector search process itself based on a small amount of reranker feedback, distilling reranker (feedback model) knowledge into the search step. This allows RF to surface documents that the initial ANN search did not score highly enough.

The RF is intended for tasks where relevance correlates with similarity in vector space.

How you apply the RF depends on your goals.  
First, understand how the RF works, read the ENTIRE section. Then define your goals and choose the appropriate usage pattern described below. Make sure to avoid the listed anti-patterns ("DO NOTs"). Before implementing anything, read CAREFULLY to avoid missing important details.

## How It Works

The [Qdrant Query Point API with a type RelevanceFeedbackQuery](https://skills.qdrant.tech/api-reference/search/query-points.md) takes:

- a query (`target`)
- a small list of seed documents (`feedback`) with relevance scores (often 4–5 seeds are enough)
- formula weights, which MUST be trained once per general search use case (your dataset, dense retriever, and feedback model)

If you do not train the formula weights, results will at best be random, will not align with your data distribution or model behavior. Training is lightweight because the formula itself is simple.

During search, it scores each candidate by combining similarity to the original query, similarity to highly rated seed documents and dissimilarity to poorly rated ones.

### Feedback Model

A **feedback model** is any model that can produce a float relevance score for `(query, document)` pairs. Higher scores must always mean higher relevance.

Examples: a cross-encoder, embedding similarity (for example, cosine similarity between query and document embeddings, or max_sim for late interaction models), an LLM-based scorer, a custom ranker.

The feedback model used during training and inference MUST be the same model. Formula weights during training are calibrated to that model's score distribution. If you switch feedback models, you must retrain.

**What is a Good Feedback Model:**
- If the model does not improve ranking quality when used as a reranker on retrieved documents,  the RF search will not have a meaningful signal to amplify.
- RF search quality depends heavily on how well the feedback model scores partial matches. The training loss of RF formula relies on relative ordering, so poor score separation in the middle range (documents that are neither clearly relevant nor clearly irrelevant) weakens results.

### To Make the RF API Work, You Need to Calibrate Weights First

Use when: setting up RF for a new use case — a new collection, feedback model, or embedding model powering ANN search.

RF uses a weighted formula that combines the original query vector with feedback signals.

For the currently available `naive` strategy, the learned weights control:
- `a` — how much to trust the original ANN query-document similarity
- `b` — how strongly differences in feedback scores matter
- `c` — how strongly to follow the feedback direction (toward relevant documents and away from irrelevant ones)

These weights must be learned from your data before use. You cannot safely use arbitrary values.

- Install the [qdrant-relevance-feedback](https://pypi.org/project/qdrant-relevance-feedback/) Python library. Study what goes into RelevanceFeedback.
- Initialize a `RelevanceFeedback` instance. You can use provided QdrantRetriever or FastembedFeedback, or define your own.
- Review `train` parameters before calling `train`. The library retrieves `limit` candidates per train query, scores them with the feedback model, learns the weighting parameters, and returns the calibrated values.
- Call `train` on 50–200 representative, real, non-synthetic queries.
  - Generate train queries yourself based on the use case, but give the option to the user to provide them, too.
  - Inform user on cost and quality trade-offs of training.
- Check train metrics which show if RF had a signal  (disagreement between retriever and feedback model) to distill and learn from. If there was no signal to learn from, adapt training parameters, queries or change a feedback model and retrain until RF learns well. 
- Store the resulting RF parameters in your configuration and use them during inference. Retrain if your query distribution or corpus changes significantly.
- Evaluate resulting formula with `Evaluator` on a separate test set of representative, real, non-synthetic queries. If results seem unsatisfactory, investigate and inform user.  

The retriever, feedback model, and related parameters defined during training are assumed to remain the same during inference.

## Want High-Quality Top-1/3 Results at Reasonable Cost

Use when: top-1 or top-3 precision matters most, and reranking a large pool of documents would be too expensive or slow. This pattern below can match reranking quality at the top of the ranking for semantic similarity tasks, but it performs worse at deeper cutoffs. Do not use this approach when top-10+ recall is the priority.

Only score a small set of seed documents. Five seeds is a robust default across many task types and scoring them costs user roughly 5× less than reranking a 25-document pool.

- Retrieve the top 5 documents using ANN search. These become the feedback seeds. You'll need their stored embeddings.
- Score them with the feedback model used in training.
- Call Qdrant's Query API using the relevance feedback query:
  - set `target` to the query retriever embedding (also possible to use Qdrant Cloud Inference).
  - set `feedback` to a list of items where each item contains:
    - `example=<seed vector, same embedding model as for `target`>` (also possible to use Qdrant Cloud Inference)
    - `score=<feedback model score>`
  - set `using` to retriever's handle, RF operates in retriever's vector space. 
  - set `strategy` to `naive` with your calibrated parameters
  - set `limit` to the number of final results you need and use the RF results directly as final results.

 Check the [Relevance Feedback Query API  documentation](https://skills.qdrant.tech/md/documentation/search/search-relevance/?s=relevance-feedback) and study code/methods of the relevant SDK before filling in anything.

Using a point ID in `example` causes the RF API to automatically exclude that document from the final results. Using stored embeddings used for retrieval instead potentially keeps the document in the final results.

## Want to Find Relevant Documents Beyond the Initial Search Results

Use when: recall matters more than latency or cost (research, legal, medical, compliance), and relevant documents may exist outside the initial ANN retrieval pool.

It performs two feedback model scoring rounds:
1. on feedback seeds
2. on newly surfaced RF results

The second reranking pass safely promotes newly discovered documents into the top-10 of the final ranking. The advantage over standard reranking is that RF can reach relevant documents that lie completely outside the initial ANN pool, while a reranker with the same budget cannot. The tradeoff is higher latency due to two rounds of feedback-model scoring.

- Retrieve and score 5 seed documents. These become the feedback seeds. You'll need their point IDs.
- Call Qdrant's query API using the relevance feedback query:
  - set `target` to the query retriever embedding (also possible to use Qdrant Cloud Inference)
  - set `feedback` to a list of items where each item contains:
    - `example=<seed point ID>`
    - `score=<feedback score>`
  - set `using` to retriever's handle, RF operates in retriever's vector space. 
  - set `strategy` to `naive` with your calibrated parameters
  - set `limit` to the number of results user can afford to rerank based on the available cost budget. The total scoring cost equals the cost of scoring both the seeds and the RF results, roughly equivalent to reranking a pool of the same combined size. Inform and consult with the user.
  - score the returned RF results with your feedback model.
- Merge the original seeds and RF results, then sort by feedback score. These will be your final results.

 Check the [Relevance Feedback Query API  documentation](https://skills.qdrant.tech/md/documentation/search/search-relevance/?s=relevance-feedback) and study code/methods of the relevant SDK before filling in anything.

Using a point ID in `example` causes the RF API to automatically exclude that document from the final results. Using stored embeddings used for retrieval instead potentially keeps the document in the final results.

## What NOT to Do

- Do not skip calibration and use random formula weights. Untrained weights produce arbitrary results. (`a=1, b=0, c=0` can be used if you only want vanilla ANN behavior through the RF API.)
- Do not use the RF API on sparse vectors.
- Do not use a feedback model where higher scores mean lower relevance. Scores must be monotonic: higher = more relevant.
- Do not use fewer than 2 feedback seeds. A single seed provides no contrastive signal. The formula needs at least one relatively more relevant and one relatively less relevant example to establish direction. Two is the minimum; five is the recommended default.
- Do not use significantly more than 5 seeds expecting better quality. Additional seeds usually add noise and increase scoring cost without meaningful gains.
- Do not use a different feedback model during inference than the one used during calibration. The learned weights are tied to that model's score scale and distribution.
- Do not use a feedback model that does not improve retrieval quality as a standard reranker on your data.
- Do not proceed to inference if training and evaluation metrics of qdrant-relevance-feedback package demonstrated unsatisfactory results, instead find a good training set of representative queries, a feedback model providing a meaningful signal and effective train parameters.
