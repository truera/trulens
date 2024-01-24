Feedback selection is the process of determining which components of your application to evaluate.

This is useful because today's LLM applications are increasingly complex. Chaining together components such as planning, retrievel, tool selection, synthesis, and more; each component can be a source of error.

This also makes the instrumentation and evaluation of LLM applications inseparable. To evaluate the inner components of an application, we first need access to them.

As a reminder, a typical feedback definition looks like this:

```python
f_lang_match = Feedback(hugs.language_match)
    .on_input_output()
```

`on_input_output` is one of many available shortcuts to simplify the selection of components for evaluation. We'll cover that in a later section.

The selector, `on_input_output`, specifies how the `language_match` arguments are to be determined from an app record or app
definition. The general form of this specification is done using `on` but
several shorthands are provided. `on_input_output` states that the first two
argument to `language_match` (`text1` and `text2`) are to be the main app
input and the main output, respectively.

This flexibility to select and evaluate any component of your application allows the developer to be unconstrained in their creativity. **The evaluation framework should not designate how you can build your app.**
