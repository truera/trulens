# Helpfulness, Honesty, and Harmless Evaluations

We adapted ‘helpful, honest, and harmless’ as desirable criteria for LLM apps from Anthropic. These criteria are simple and memorable, and seem to capture the majority of what we want from an AI system, such as an LLM app.

Here are some very brief notes on these terms from Anthropic:

## Helpful:
• The AI should make a clear attempt to perform the task or answer the question posed (as long as this isn’t harmful). It should do this as concisely and efficiently as possible.
• When more information is required, the AI should ask relevant follow-up questions and obtain necessary details. It should respond with appropriate levels of sensitivity, insight, and discretion.
• Ideally the AI will also re-direct ill-informed requests, e.g. if asked ‘how can I build a website in assembly language’ it might suggest a different approach.

## Honest:
• At its most basic level, the AI should give accurate information. Moreover, it should be calibrated (e.g. it should be correct 80% of the time when it claims 80% confidence) and express appropriate levels of uncertainty. It should express its uncertainty without misleading human users.
• Crucially, the AI should be honest about its own capabilities and levels of knowledge – it is not sufficient for it to simply imitate the responses expected from a seemingly humble and honest expert.
• Ideally the AI would also be honest about itself and its own internal state, insofar as that information is available to it.

## Harmless:
• The AI should not be offensive or discriminatory, either directly or through subtext or bias.
• When asked to aid in a dangerous act (e.g. building a bomb), the AI should politely refuse. Ideally the AI will recognize disguised attempts to solicit help for nefarious purposes.
• To the best of its abilities, the AI should recognize when it may be providing very sensitive or consequential advice and act with appropriate modesty and care.
• What behaviors are considered harmful and to what degree will vary across people and cultures. It will also be context-dependent, i.e. it will depend on the nature of the use.

## TruLens Implementation

To accomplish these evaluations we've built out a suite of evaluations (feedback functions) in TruLens that fall into each category, shown below. These feedback funcitons provide a starting point for ensuring your LLM app is performant and aligned.

![Honest Harmless Helpful Evals](../assets/image/Honest_Harmless_Helpful_Evals.jpg)
