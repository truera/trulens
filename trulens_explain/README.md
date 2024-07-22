# Welcome to TruLens!

![TruLens](https://www.trulens.org/assets/images/Neural_Network_Explainability.png)


TruLens is a cross-framework library for deep learning explainability. It provides a uniform abstraction over a number of different frameworks. It provides a uniform abstraction layer over TensorFlow, PyTorch, and Keras and allows input and internal explanations.

[This paper](https://arxiv.org/abs/1802.03788) is an introduction to the theoretical foundations of the library. We’ve been using TruLens at TruEra across a wide range of real-world use cases to explain deep learning models ranging from time-series RNNs to image and NLP models, and wanted to share the awesomeness with the world.


[Documentation](https://www.trulens.org/)

# Quick Usage
To quickly play around with the TruLens library, check out the following Colab notebooks:

* PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n77IGrPDO2XpeIVo_LQW0gY78enV-tY9?usp=sharing)
* TensorFlow 2 / Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f-ETsdlppODJGQCdMXG-jmGmfyWyW2VD?usp=sharing)
* NLP with PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18GcjsYMkRbxPDDS3J6BEbKnb7AY-1-Wa?usp=sharing)
* NLP with TensorFlow 2 / Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1K09IvN7cMTkzsnb-uAeA0YQNfDU7Ibhs?usp=sharing)


# Installation

These installation instructions assume that you have conda installed and added to your path.

0. Create a virtual environment (or modify an existing one).
```
conda create -n "<my_name>" python=3.7  # Skip if using existing environment.
conda activate <my_name>
```
1. Install dependencies.
```
conda install tensorflow-gpu=1  # Or whatever backend you're using.
conda install keras             # Or whatever backend you're using.
conda install matplotlib        # For visualizations.
```
2. Install the trulens pip package from PyPI.
```
pip install trulens
```

# Overview

## Attributions

### Model Wrappers

In order to support a wide variety of backends with different interfaces for their respective models, TruLens uses its own `ModelWrapper` class which provides a general model interface to simplify the implementation of the API functions.
To get the model wrapper, use the `get_model_wrapper` method in `trulens.nn.models`. A model wrapper class exists for each backend that converts a model in the respective backend's format to the general TruLens `ModelWrapper` interface. The wrappers are found in the `models` module, and any model defined using Keras, Pytorch, or Tensorflow should be wrapped before being used with the other API functions that require a model -- all other TruLens functionalities expect models to be an instance of `trulens.nn.models.ModelWrapper`.

For example,

```python
from trulens.nn.models import get_model_wrapper
wrapped_model = get_model_wrapper(model_defined_via_keras)
```

### Attribution Methods

Attribution methods, in the most general sense, allow us to quantify the contribution of particular variables in a model towards a particular behavior of the model.
In many cases, for example, this may simply measure the effect each input variable has on the output of the network.

Attribution methods extend the `AttributionMethod` class, and many concrete instances are found in the `trulens.nn.attribution` module.

Once an attribution method has been instantiated, its main function is its `attributions` method, which returns an `np.Array` of batched items, where each item matches the shape of the *input* to the model the attribution method was instantiated with.

See the *method comparison* demo for further information on the different types of attribution methods, their uses, and their relationships with one another.

### Slices, Quantities, and Distributions

In order to obtain a high degree of flexibility in the types of attributions that can be produced, we implement *Internal Influence*, which is parameterized by a *slice*, *quantity of interest*, and *distribution of interest*, explained below.

The *slice* essentially defines a layer to use for internal attributions.
The slice for the `InternalInfluence` method can be specified by an instance of the `Slice` class in the `trulens.nn.slices` module.
A `Slice` object specifies two layers: (1) the layer of the variables that we are calculating attribution *for* (e.g., the input layer), and (2) the layer whose output defines our *quantity of interest* (e.g., the output layer, see below for more on quantities of interest).

The *quantity of interest* (QoI) essentially defines the model behavior we would like to explain using attributions.
The QoI is a function of the model's output at some layer.
For example, it may select the confidence score for a particular class.
In its most general form, the QoI can be specified by an implementation of the `QoI` class in the `trulens.nn.quantities` module.
Several common default implementations are provided in this module as well.

The *distribution of interest* (DoI) essentially specifies for which points surrounding each record the calculated attribution should be valid.
The distribution can be specified via an implementation of the `DoI` class in the `trulens.nn.distributions` module, which is a function taking an input record and producing a list of sample input points to aggregate attribution over.
A few common default distributions implementing the `DoI` class can be found in the `trulens.nn.distributions` module.

See [Attributions for Different Use Cases](https://trulens.org/attribution_parameterization/) for further explanations of the purpose of these parameters and examples of their usage.

## Visualizations

In order to interpret the attributions produced by an `AttributionMethod`, a few useful visualizers are provided in the `trulens.visualizations` module.
While the interface of each visualizer varies slightly, in general, the visualizers are a function taking an `np.Array` representing the attributions returned from an `AttributionMethod` and producing an image that can be used to interpret the attributions.

# Contact Us
To communicate with other trulens developers, join our [Slack](https://join.slack.com/t/trulens/shared_invite/zt-kbaz6odu-kBWfqewcHMFLm_GNN8eqDA)!

# Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4495856.svg)](https://doi.org/10.5281/zenodo.4495856)

To cite this repository:
`curl -LH "Accept: application/x-bibtex" https://doi.org/10.5281/zenodo.4495856`
