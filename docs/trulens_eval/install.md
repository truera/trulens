## Getting access to TruLens

These installation instructions assume that you have conda installed and added to your path.

1. Create a virtual environment (or modify an existing one).
```
conda create -n "<my_name>" python=3  # Skip if using existing environment.
conda activate <my_name>
```

2. [Pip installation] Install the trulens-eval pip package.
```
pip install trulens-eval
```

3. [Local installation] If you would like to develop or modify trulens, you can download the source code by cloning the trulens repo.
```
git clone https://github.com/truera/trulens.git
```

4. [Locall installation] Install the trulens repo.
```
cd trulens/trulens_eval
pip install -e .
```


