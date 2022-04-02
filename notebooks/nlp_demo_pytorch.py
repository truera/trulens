import sys

# Use this if running this notebook from within its place in the truera repository.
sys.path.insert(0, "..")

import torch

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

# Wrap all of the necessary components.
class TwitterSentiment:
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"

    device = 'cpu'
    # Can also use cuda if available:
    # device = 'cuda:0'

    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = ['negative', 'neutral', 'positive']

    NEGATIVE = labels.index('negative')
    NEUTRAL = labels.index('neutral')
    POSITIVE = labels.index('positive')

task = TwitterSentiment()

sentences = ["I'm so happy!", "I'm so sad!", "I cannot tell whether I should be happy or sad!", "meh"]

# Input sentences need to be tokenized first.

inputs = task.tokenizer(sentences, padding=True, return_tensors="pt").to(task.device) # pt refers to pytorch tensor

# The tokenizer gives us vocabulary indexes for each input token (in this case,
# words and some word parts like the "'m" part of "I'm" are tokens).

print(inputs)

# Decode helps inspecting the tokenization produced:

print(task.tokenizer.batch_decode(torch.flatten(inputs['input_ids'])))
# Normally decode would give us a single string for each sentence but we would
# not be able to see some of the non-word tokens there. Flattening first gives
# us a string for each input_id.

outputs = task.model(**inputs)

print(outputs)

# From logits we can extract the most likely class for each sentence and its readable label.

predictions = [task.labels[i] for i in outputs.logits.argmax(axis=1)]

for sentence, logits, prediction in zip(sentences, outputs.logits, predictions):
    print(logits.to('cpu').detach().numpy(), prediction, sentence)

from trulens.nn.models import get_model_wrapper
from trulens.nn.quantities import ClassQoI
from trulens.nn.attribution import IntegratedGradients
from trulens.nn.attribution import Cut, OutputCut
from trulens.utils.typing import ModelInputs

task.wrapper = get_model_wrapper(task.model, input_shape=(None, task.tokenizer.model_max_length), device=task.device)

task.wrapper.print_layer_names()

infl_max = IntegratedGradients(
    model = task.wrapper,
    doi_cut=Cut('roberta_embeddings_word_embeddings'),
    qoi_cut=OutputCut(accessor=lambda o: o['logits'])
)

# Alternatively we can look at a particular class:

infl_positive = IntegratedGradients(
    model = task.wrapper,
    doi_cut=Cut('roberta_embeddings_word_embeddings'),
    qoi=ClassQoI(task.POSITIVE),
    qoi_cut=OutputCut(accessor=lambda o: o['logits'])
)

attrs = infl_max.attributions(**inputs)

for token_ids, token_attr in zip(inputs['input_ids'], attrs):
    for token_id, token_attr in zip(token_ids, token_attr):
        # Not that each `word_attr` has a magnitude for each of the embedding
        # dimensions, of which there are many. We aggregate them for easier
        # interpretation and display.
        attr = token_attr.sum()

        word = task.tokenizer.decode(token_id)

        print(f"{word}({attr:0.3f})", end=' ')

    print()

from trulens.visualizations import NLP, HTML

h = HTML()

V = NLP(
    wrapper=task.wrapper,
    output=h,
    labels=task.labels,
    decode=lambda x: task.tokenizer.decode(x),
    tokenize=lambda sentences: ModelInputs(kwargs=task.tokenizer(sentences, padding=True, return_tensors='pt')).map(lambda t: t.to(task.device)),
    # huggingface models can take as input the keyword args as per produced by their tokenizers.

    input_accessor=lambda x: x.kwargs['input_ids'],
    # for huggingface models, input/token ids are under input_ids key in the input dictionary

    output_accessor=lambda x: x['logits'],
    # and logits under 'logits' key in the output dictionary

    hidden_tokens=set([task.tokenizer.pad_token_id])
    # do not display these tokens
)

print("QOI = MAX PREDICTION")
html = V.token_attribution(sentences, infl_max)
print(html)
h.open(html)

print("QOI = POSITIVE")
html = V.token_attribution(sentences, infl_positive)
print(html)
h.open(html)