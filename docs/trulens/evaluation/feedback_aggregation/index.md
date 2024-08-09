# Feedback Aggregation

For cases where argument specification names more than one value as an input,
aggregation can be used.

Consider this feedback example:

```python
# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve.args.query)
    .on(Select.RecordCalls.retrieve.rets)
    .aggregate(np.mean)
)
```

The last line `aggregate(numpy.min)` specifies how feedback outputs are to be aggregated.
This only applies to cases where the argument specification names more than one value
for an input. The second specification, for `statement` was of this type.

The input to `aggregate` must be a method which can be imported globally. This function
is called on the `float` results of feedback function evaluations to produce a single float.

The default is `numpy.mean`.
