## Attribution Parameterization 

Attributions for different models and use cases can range from simple to more complex. This will help see how to set the various parameters to get what you need.

### Basic Definitions and Terminology

**What is a tensor?**
A tensor is a multidimensional object that can be model inputs, or layer activations.

**What is a layer?**
A layer is a set of neurons that can be thought of as a function on input tensors. Layer inputs are tensors. Layer outputs are modified tensors.

**What are anchors?**
Anchors are ways of specifying which tensors you want. You may want the input tensor of a layer, or the output tensor of a layer. 

Example use case: Say you have a concat layer and you want to explain the 2 concatenated tensors. The concat operation is not usually a layer tracked by the model. If you try the 'in' anchor of the layer after the operation, you get a single tensor with all the information you need.

**What is a Quantity of Interest (QoI)?**
A QoI is a scalar number that is being explained. 

Eg: With saliency maps, you get `dx/dy` (i.e the effect of input on output). `y` in this case is the QoI scalar. It is usually the output of a neuron, but could be a sum of multiple neurons.

**What is an attribution?**
An attribution is a numerical number associated with every element in a tensor that explains a QoI. 

Eg: With saliency maps, you get `dx/dy`. `x` is the associated tensor. The entirety of `dx/dy` is the explanation.

**What are cuts?**
Cuts are tensors that cut a network into two parts. They are composed of a layer and an anchor.

**What are slices?**
Slices are two cuts leaving a `slice` of the network. The attribution will be on the first cut, explaining the QoI on the second cut of the slice.

Eg: With saliency maps, the trulens slice would be AttributionCut:`Cut(x)` to QoICut:`Cut(y)` denoted by `Slice(Cut(x),Cut(y))`

### How to use Trulens?

This section will cover different use cases from the most basic to the most complex.

#### Case 1: Input-Output cut (Basic configuration)

**Use case:** Explain the input given the output.  
**Cuts needed:** Trulens Defaults.  
**Attribution Cut** (The tensor we would like to assign importance) → InputCut aka model args/kwargs  
**QoI Cut** (The tensor that we are interested to explain) → OutputCut
  
#### Case 2: The QoI Cut

Now suppose you want to explain some internal layer’s output (intermediate output)  
i.e how the input is affecting the output at some intermediate layer.

**Use case:** Explain something that isn't the default model output.  
Eg: If you want to explain a logit layer instead of the probit layer (the final layer)  
**Cuts needed:** As you want to explain something different than the default output, you need to change the QoI from the default to the layer that you are interested.  
**Attribution Cut** → InputCut  
**QoI Cut** → Your logit layer, anchor:'out'
 
#### Case 3: The Attribution Cut
Now suppose you want to know the attribution of some internal layer on the final output. 

**Use cases:** 

* As a preprocessing step, you drop a feature, so do not need attributions on that.
* For torch models, model inputs are not tensors. so you'd want the in anchor of the first layer  

**Cuts needed:** As you want to know the affect of some other layer rather than the input layer, you need to customize the attribution cut.  
**Model inputs** → InputCut  
**Attribution Cut** → Your attribution layer (The layer you want to assign importance/attributions with respect to output), anchor:'in'  
**QoI Cut** → OutputCut
 
#### Case 4: The DoI Cut / Explanation flexibility

Usually, we explain the output with respect to each point in the input. Now, suppose you want to explain using an aggregate over samples of points  

**Use case:** You want to do integrated gradients, gradcam, shapley values instead of saliency maps. These only differ by sampling strategies.  
Eg: Integrated gradients is a sample from a straight line from a baseline to a value.  
**Cuts needed:** Define a DoI that samples from the default attribution cut  
**Model inputs** → InputCut  
**DoI/Attribution Cut** → Your baseline/DoI/attribution layer, anchor:'in'  
**QoI Cut** → OutputCut
 
#### Case 5: Internal explanations

**Use case:** You want to explain an internal layer. Things like integrated gradients are a DoI on the baseline to the value, but it is on the layer the baseline is defined.
If you want to explain an internal layer, you do not move the DoI layer.  
**Cuts needed:** Attribution layer different from DoI  
**Model inputs** → InputCut  
**DoI Cut** → Your baseline/DoI layer, anchor:'in'  
**Attribution Cut** → Your internal attribution layer, anchor:'out' or 'in'  
**QoI Cut** → OutputCut
 
#### Case 6: Your baseline happens at a different layer than your sampling.

**Use Case:** in NLP, baselines are tokens, but the interpolation is on the embedding layer  
**Cuts needed:** Baseline different from DoI  
**Model inputs** → InputCut  
**Baseline Cut** →  tokens, anchor:'out'  
**DoI/Attribution Cut** → embeddings, anchor:'out'  
**QoI Cut** → OutputCut
 
#### Case 7: Putting it together - The most complex one one we can do with our Trulens

**Use Case:** Internal layer explanations of NLP, on the logit layer of a model with probit outputs  
**Model inputs** → InputCut  
**Baseline Cut** → tokens, anchor:'out'  
**DoI Cut** → embeddings, anchor:'out'  
**Attribution Cut** → Internal layer, anchor:'out'  
**QoI Cut** → logit layer, anchor:'out'
 
### Summary

**InputCut** is model args / kwargs  
**OutputCut** is the model output

**Baseline Cut** is the tensor associated with the Integrated gradients baseline. Can be the InputCut or later.  
**DoI Cut** is the tensor associated with explanation sampling. Can be the BaselineCut or later  
**Attribution Cut** is the tensor that should be explained. Can be the DoICut or later  
**QoI Cut** is what is being explained with a QoI. Must be after the AttributionCut  
