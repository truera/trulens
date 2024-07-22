## Attribution Parameterization

Attributions for different models and use cases can range from simple to more complex. This page provides guidelines on how to set various attribution parameters to achieve your LLM explainability goals.

### Basic Definitions and Terminology

**What is a tensor?**
A tensor is a multidimensional object that can be model inputs, or layer activations.

**What is a layer?**
A layer is a set of neurons that can be thought of as a function on input tensors. Layer inputs are tensors. Layer outputs are modified tensors.

**What are anchors?**
Anchors are ways of specifying which tensors you want. You may want the input tensor of a layer, or the output tensor of a layer.

> E.g. Say you have a concat layer and you want to explain the 2 concatenated tensors. The concat operation is not usually a layer tracked by the model. If you try the 'in' anchor of the layer after the operation, you get a single tensor with all the information you need.

**What is a Quantity of Interest (QoI)?**
A QoI is a scalar number that is being explained.

> E.g. With saliency maps, you get `dx/dy` (i.e. the effect of input on output). `y` in this case is the QoI scalar. It is usually the output of a neuron, but could be a sum of multiple neurons.

**What is an attribution?**
An attribution is a numerical value associated with every element in a tensor that explains a QoI.

> E.g. With saliency maps, you get `dx/dy`. `x` is the associated tensor. The entirety of `dx/dy` is the explanation.

**What are cuts?**
Cuts are tensors that cut a network into two parts. They are composed of a layer and an anchor.

**What are slices?**
Slices are two cuts leaving a `slice` of the network. The attribution will be on the first cut, explaining the QoI on the second cut of the slice.

> E.g. With saliency maps, the TruLens slice would be AttributionCut: `Cut(x)` to QoICut: `Cut(y)`, denoted by `Slice(Cut(x),Cut(y))`.

### How to use TruLens?

This section will cover different use cases from the most basic to the most complex. For the following use cases, it may help to refer to [Summary](#summary).

#### Case 1: Input-Output cut (Basic configuration)

**Use case:** Explain the input given the output.
**Cuts needed:** TruLens defaults.
**Attribution Cut** (The tensor we would like to assign importance) → InputCut (model args / kwargs)
**QoI Cut** (The tensor that we are interested to explain) → OutputCut

#### Case 2: The QoI Cut

Now suppose you want to explain some internal (intermediate) layer’s output (i.e. how the input is affecting the output at some intermediate layer).

**Use case:** Explain something that isn't the default model output.

> E.g. If you want to explain a logit layer instead of the probit (final) layer.

**Cuts needed:** As you want to explain something different than the default output, you need to change the QoI from the default to the layer that you are interested.
**Attribution Cut** → InputCut
**QoI Cut** → Your logit layer, anchor:'out'

#### Case 3: The Attribution Cut
Now suppose you want to know the attribution of some internal layer on the final output.

**Use cases:**

* As a preprocessing step, you drop a feature, so do not need attributions on that.
* For PyTorch models, model inputs are not tensors, so you'd want the 'in' anchor of the first layer.

**Cuts needed:** As you want to know the affect of some other layer rather than the input layer, you need to customize the attribution cut.
**Model inputs** → InputCut
**Attribution Cut** → Your attribution layer (The layer you want to assign importance/attributions with respect to output), anchor:'in'
**QoI Cut** → OutputCut

### Advanced Use Cases
For the following use cases, it may help to refer to [Advanced Definitions](#advanced-definitions).

#### Case 4: The Distribution of Interest (DoI) Cut / Explanation flexibility

Usually, we explain the output with respect to each point in the input. All cases up to now were using a default called `PointDoI`. Now, suppose you want to explain using an aggregate over samples of points.

**Use case:** You want to perform approaches like Integrated Gradients, Grad-CAM, Shapley values instead of saliency maps. These only differ by sampling strategies.

> E.g. Integrated Gradients is a sample from a straight line from a baseline to a value.

**Cuts needed:** Define a DoI that samples from the default attribution cut.
**Model inputs** → InputCut
**DoI/Attribution Cut** → Your baseline/DoI/attribution layer, anchor:'in'
**QoI Cut** → OutputCut

#### Case 5: Internal explanations

**Use case:** You want to explain an internal layer. Methods like Integrated Gradients are a DoI on the baseline to the value, but it is located on the layer the baseline is defined.
If you want to explain an internal layer, you do not move the DoI layer.
**Cuts needed:** Attribution layer different from DoI.
**Model inputs** → InputCut
**DoI Cut** → Your baseline/DoI layer, anchor:'in'
**Attribution Cut** → Your internal attribution layer, anchor:'out' or 'in'
**QoI Cut** → OutputCut

#### Case 6: Your baseline happens at a different layer than your sampling.

**Use Case:** in NLP, baselines are tokens, but the interpolation is on the embedding layer.
**Cuts needed:** Baseline different from DoI.
**Model inputs** → InputCut
**Baseline Cut** → Tokens, anchor:'out'
**DoI/Attribution Cut** → Embeddings, anchor:'out'
**QoI Cut** → OutputCut

#### Case 7: Putting it together - The most complex case we can perform with TruLens

**Use Case:** Internal layer explanations of NLP, on the logit layer of a model with probit outputs.
**Model inputs** → InputCut
**Baseline Cut** → Tokens, anchor:'out'
**DoI Cut** → Embeddings, anchor:'out'
**Attribution Cut** → Internal layer, anchor:'out'
**QoI Cut** → Logit layer, anchor:'out'

### Summary

**InputCut** is model args / kwargs.
**OutputCut** is the model output.

**Baseline Cut** is the tensor associated with the Integrated Gradients baseline. Can be the InputCut or later.
**DoI Cut** is the tensor associated with explanation sampling. Can be the BaselineCut or later.
**Attribution Cut** is the tensor that should be explained. Can be the DoICut or later.
**QoI Cut** is what is being explained with a QoI. Must be after the AttributionCut.

### Advanced Definitions

**What is a Distribution of Interest (DoI)?**

The distribution of interest is a concept of aggregating attributions over a sample or distribution.

* Grad-CAM ([Paper](https://ieeexplore.ieee.org/document/8237336), [GitHub](https://github.com/jacobgil/pytorch-grad-cam), [Docs](https://jacobgil.github.io/pytorch-gradcam-book/introduction.html)) does this over a Gaussian distribution of inputs.
* Shapley values ([GitHub](https://github.com/slundberg/shap), [Docs](https://shap.readthedocs.io/en/latest/)) do this over different background data.
* Integrated Gradients ([Paper](https://arxiv.org/abs/1703.01365), [Tutorial](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients)) do this over an interpolation from a baseline to the input.

***How does this relate to the Attribution Cut?***

The sample or distributions are taken at a place that is humanly considered the input, even if this differs from the programmatic model input.

For attributions, all parts of a network can have an attribution towards the QoI. The most common use case is to explain the tensors that are also humanly considered the input (which is where the DoI occurs).

***How does this relate to the Baseline Cut?***

The Baseline Cut is only applicable to the Integrated Gradients method. It is also only needed when there is no mathematical way to interpolate the baseline to the input.

> E.g. if the input is `'Hello'`, but the baseline is a `'[MASK]'` token, we cannot interpolate that. We define the baseline at the token layer, but interpolate on a numeric layer like the embeddings.
