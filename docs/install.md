## Getting access to TruLens

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

2. Clone the trulens repo.
```
git clone https://github.com/truera/trulens.git
```

3. Install the trulens repo.
```
cd trulens
pip install -e .
```


