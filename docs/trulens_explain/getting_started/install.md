# Getting access to TruLens Explain

These installation instructions assume that you have conda installed and added to your path.

1. Create a virtual environment (or modify an existing one).

    ```bash
    conda create -n "<my_name>" python=3.7  # Skip if using existing environment.
    conda activate <my_name>
    ```

2. Install dependencies.

    ```bash
    conda install tensorflow-gpu=1  # Or whatever backend you're using.
    conda install keras             # Or whatever backend you're using.
    conda install matplotlib        # For visualizations.
    ```

3. [Pip installation] Install the trulens pip package from PyPI.

    ```bash
    pip install trulens
    ```

4. [Local installation] If you would like to develop or modify TruLens, you can
   download the source code by cloning the TruLens repo.

    ```bash
    git clone https://github.com/truera/trulens.git
    ```

5. [Local installation] Install the TruLens repo.

    ```bash
    cd trulens_explain
    pip install -e .
    ```
