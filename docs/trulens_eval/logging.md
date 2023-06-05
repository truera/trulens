# Logging

## Automatic Logging

The simplest method for logging with TruLens is by wrapping with TruChain and including the tru argument, as shown in the quickstart.

This is done like so:

```python
truchain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    tru=tru
)
truchain("This will be automatically logged.")
```

Feedback functions can also be logged automatically by providing them in a list to the feedbacks arg.

```python
truchain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match], # feedback functions
    tru=tru
)
truchain("This will be automatically logged.")
```

## Manual Logging

### Wrap with TruChain to instrument your chain
```python
tc = tru_chain.TruChain(chain, chain_id='Chain1_ChatApplication)
```

### Set up logging and instrumentation

Making the first call to your wrapped LLM Application will now also produce a log or "record" of the chain execution.
```python
prompt_input = 'que hora es?'
gpt3_response, record = tc(prompt_input)
```

We can log the records but first we need to log the chain itself.

```python
tru.add_chain(chain_json=truchain.json)
```

Then we can log the record:
```python
tru.add_record(
    prompt=prompt_input, # prompt input
    response=gpt3_response['text'], # LLM response
    record_json=record # record is returned by the TruChain wrapper
)
```

### Evaluate Quality

Following the request to your app, you can then evaluate LLM quality using feedback functions. This is completed in a sequential call to minimize latency for your application, and evaluations will also be logged to your local machine.

To get feedback on the quality of your LLM, you can use any of the provided feedback functions or add your own.

To assess your LLM quality, you can provide the feedback functions to `tru.run_feedback()` in a list provided to `feedback_functions`.

```python
feedback_results = tru.run_feedback_functions(
    record_json=record,
    feedback_functions=[f_lang_match]
)
display(feedback_results)
```

After capturing feedback, you can then log it to your local database.
```python
tru.add_feedback(feedback_results)
```

### Out-of-band Feedback evaluation

In the above example, the feedback function evaluation is done in the same process as the chain evaluation. The alternative approach is the use the provided persistent evaluator started via `tru.start_deferred_feedback_evaluator`. Then specify the `feedback_mode` for `TruChain` as `deferred` to let the evaluator handle the feedback functions.

For demonstration purposes, we start the evaluator here but it can be started in another process.
```python
truchain: TruChain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru,
    feedback_mode="deferred"
)

tru.start_evaluator()
truchain("This will be logged by deferred evaluator.")
tru.stop_evaluator()
```
