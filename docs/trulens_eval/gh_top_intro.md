# Welcome to TruLens!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

TruLens provides a set of tools for developing and monitoring neural nets, including large language models. This includes both tools for evaluation of LLMs and LLM-based applications with TruLens-Eval and deep learning explainability with TruLens-Explain. TruLens-Eval and TruLens-Explain are housed in separate packages and can be used independently.

The best way to support TruLens is to give us a ⭐ and join our [slack community](https://communityinviter.com/apps/aiqualityforum/josh)!

## TruLens-Eval

**TruLens-Eval** contains instrumentation and evaluation tools for large language model (LLM) based applications. It supports the iterative development and monitoring of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine. Importantly, it also gives you the tools you need to evaluate the quality of your LLM-based applications.

![Architecture Diagram](https://www.trulens.org/Assets/image/TruLens_Architecture.png)

### Get going with TruLens-Eval

Install trulens-eval from PyPI.

```bash
pip install trulens-eval
```
