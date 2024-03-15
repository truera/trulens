# Generating Test Cases

Generating a sufficient test set for evaluating an app is an early change in the
development phase.

TruLens allows you to generate a test set of a specified breadth and depth,
tailored to your app and data. Resulting test set will be a list of test prompts
of length `depth`, for `breadth` categories of prompts. Resulting test set will
be made up of `breadth` X `depth` prompts organized by prompt category.

Example:

```python
from trulens_eval.generate_test_set import GenerateTestSet

test = GenerateTestSet(app_callable = rag_chain.invoke)
test_set = test.generate_test_set(
  test_breadth = 3,
  test_depth = 2
)
test_set
```

Returns:

```python
{'Code implementation': [
  'What are the steps to follow when implementing code based on the provided instructions?',
  'What is the required format for each file when outputting the content, including all code?'
  ],
 'Short term memory limitations': [
  'What is the capacity of short-term memory and how long does it last?',
  'What are the two subtypes of long-term memory and what types of information do they store?'
  ],
 'Planning and task decomposition challenges': [
  'What are the challenges faced by LLMs in adjusting plans when encountering unexpected errors during long-term planning?',
  'How does Tree of Thoughts extend the Chain of Thought technique for task decomposition and what search processes can be used in this approach?'
  ]
}
```

Optionally, you can also provide a list of examples (few-shot) to guide the LLM
app to a particular type of question.

Example:

```python
examples = [
  "What is sensory memory?",
  "How much information can be stored in short term memory?"
]

fewshot_test_set = test.generate_test_set(
  test_breadth = 3,
  test_depth = 2,
  examples = examples
)
fewshot_test_set
```

Returns:

```python
{'Code implementation': [
  'What are the subcategories of sensory memory?',
  'What is the capacity of short-term memory according to Miller (1956)?'
  ],
 'Short term memory limitations': [
  'What is the duration of sensory memory?',
  'What are the limitations of short-term memory in terms of context capacity?'
  ],
 'Planning and task decomposition challenges': [
  'How long does sensory memory typically last?',
  'What are the challenges in long-term planning and task decomposition?'
  ]
}
```

In combination with record metadata logging, this gives you the ability to
understand the performance of your application across different prompt
categories.

```python
with tru_recorder as recording:
    for category in test_set:
        recording.record_metadata=dict(prompt_category=category)
        test_prompts = test_set[category]
        for test_prompt in test_prompts:
            llm_response = rag_chain.invoke(test_prompt)
```
