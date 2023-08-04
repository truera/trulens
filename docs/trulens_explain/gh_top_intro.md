## TruLens-Explain

**TruLens-Explain** is a cross-framework library for deep learning explainability. It provides a uniform abstraction over a number of different frameworks. It provides a uniform abstraction layer over TensorFlow, Pytorch, and Keras and allows input and internal explanations.

### Installation and Setup

These installation instructions assume that you have conda installed and added to your path.

0. Create a virtual environment (or modify an existing one).
```bash
conda create -n "<my_name>" python=3  # Skip if using existing environment.
conda activate <my_name>
```
 
1. Install dependencies.
```bash
conda install tensorflow-gpu=1  # Or whatever backend you're using.
conda install keras             # Or whatever backend you're using.
conda install matplotlib        # For visualizations.
```

2. [Pip installation] Install the trulens pip package from PyPI.
```bash
pip install trulens
```

### Quick Usage

To quickly play around with the TruLens library, check out the following Colab notebooks:

* PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n77IGrPDO2XpeIVo_LQW0gY78enV-tY9?usp=sharing)
* TensorFlow 2 / Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f-ETsdlppODJGQCdMXG-jmGmfyWyW2VD?usp=sharing)

For more information, see [TruLens-Explain Documentation](https://www.trulens.org/trulens_explain/quickstart/).
