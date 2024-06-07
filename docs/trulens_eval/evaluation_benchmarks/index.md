# Evaluation Benchmarks

TruLens relies on feedback functions to score the performance of LLM apps, which are implemented across a variety of LLMs and smaller models. The numerical scoring scheme adopted by _TruLens_' feedback functions is intuitive for generating aggregated results from eval runs that are easy to interpret and visualize across different applications of interest. However, it begs the question how trustworthy these scores actually are, given they are at their core next-token-prediction-style generation from meticulously designed prompts.

Consequently, these feedback functions face typical large language model (LLM) challenges in rigorous production environments, including prompt sensitivity and non-determinism, especially when incorporating Mixture-of-Experts and model-as-a-service solutions like those from _OpenAI_. Drawing inspiration from works on [Judging LLM-as-a-Judge](https://arxiv.org/pdf/2306.05685), we outline findings from our analysis of feedback function performance against task-aligned benchmark data. To accomplish this, we first need to align feedback function tasks to relevant benchmarks in order to gain access to large scale ground truth data for the feedback functions. We then are able to easily compute metrics such as MAE (mean absolute error) across a variety of implementations and models.

Here, we'll outline the primary findings from these analyses on critical feedback functions across a variety of implementations. We will continue to expand this benchmarking over time.

## Context Relevance

Observing that many information retrieval (IR) benchmarks, such as those found at https://huggingface.co/datasets/ms_marco/viewer/v2.1, use binary labels, we propose to frame the problem of evaluating context relevance tasks as evaluating a recommender system. In essence, we argue that the relative importance or ranking based on the score assignments is all you need to achieve meta-evaluation against human golden sets. The intuition is that it is a sufficient proxy to trustworthiness if feedback functions demonstrate discriminative capabilities that reliably and consistently assign items, be they context chunks or generated responses, with weights and ordering closely mirroring human preferences.

Here, we leverage the 500 random sampled rows from [MS MARCO](https://arxiv.org/abs/1611.09268) dataset for benchmarking context relevance.

### Performance on MS MARCO

| Feedback Function Base Model | NDCG | nDCG | Recall@K | Precision@1 |
| --- | --- | --- | --- | --- |
| GPT 3.5 Turbo | XX | XX | XX | XX |
| GPT 4 Turbo | XX | XX | XX | XX |
| GPT 4o | XX | XX | XX | XX |
| Arctic Instruct | XX | XX | XX | XX |
| Llama 3 8B | XX | XX | XX | XX |
| Mistral 7B | XX | XX | XX | XX |

[Benchmarking code](context_relevance_benchmark.ipynb)

## Groundedness

Observing that many summarization benchmarks, such as those found at [SummEval](https://arxiv.org/abs/2007.12626), use human annotation of numerical scores, we propose to frame the problem of evaluating groundedness tasks as evaluating a summarization system. In particular, we generate test cases from [SummEval](https://arxiv.org/abs/2007.12626).

SummEval is one of the datasets dedicated to automated evaluations on summarization tasks, which are closely related to the groundedness evaluation in RAG with the retrieved context (i.e. the source) and response (i.e. the summary). It contains human annotation of numerical score (**1** to **5**) comprised of scoring from 3 human expert annotators and 5 croweded-sourced annotators. There are 16 models being used for generation in total for 100 paragraphs in the test set, so there are a total of 16,000 machine-generated summaries. Each paragraph also has several human-written summaries for comparative analysis. 

For evaluating groundedness feedback functions, we compute the annotated "consistency" scores, a measure of whether the summarized response is factually consisntent with the source texts and hence can be used as a proxy to evaluate groundedness in our RAG triad, and normalized to **0** to **1** score as our **expected_score** and to match the output of feedback functions.

| Feedback Function Base Model | MAE | Latency | Total Cost |
| --- | --- | --- | --- |
| GPT 3.5 Turbo | XX | XX | XX |
| GPT 4 Turbo | XX | XX | XX |
| GPT 4o | XX | XX | XX |
| Arctic Instruct | XX | XX | XX |
| Llama 3 8B | XX | XX | XX |
| Mistral 7B | XX | XX | XX |

[Benchmarking code](groundedness_benchmark.ipynb)

## Comprehensiveness

This notebook follows an evaluation of a set of test cases generated from human
annotated datasets. In particular, we generate test cases from
[MeetingBank](https://arxiv.org/abs/2305.17529) to evaluate our
comprehensiveness feedback function.

MeetingBank is one of the datasets dedicated to automated evaluations on
summarization tasks, which are closely related to the comprehensiveness
evaluation in RAG with the retrieved context (i.e. the source) and response
(i.e. the summary). It contains human annotation of numerical score (**1** to
**5**). 

For evaluating comprehensiveness feedback functions, we compute the annotated
"informativeness" scores, a measure of how well  the summaries capture all the
main points of the meeting segment. A good summary should contain all and only
the important information of the source., and normalized to **0** to **1** score
as our **expected_score** and to match the output of feedback functions.

