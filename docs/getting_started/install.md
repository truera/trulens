# 🔨 Installation

!!! info
    TruLens now operates on OpenTelemetry traces. [Read more](../blog/posts/trulens_otel.md).

These installation instructions assume that you have conda installed and added
to your path.

1. Create a virtual environment (or modify an existing one).

    ```bash
    conda create -n "<my_name>" python=3  # Skip if using existing environment.
    conda activate <my_name>
    ```

2. [Pip installation] Install the trulens pip package from PyPI.

    ```bash
    pip install trulens
    ```

3. [Local installation] If you would like to develop or modify TruLens, you can
   download the source code by cloning the TruLens repo.

    ```bash
    git clone https://github.com/truera/trulens.git
    ```

4. [Local installation] Install the TruLens repo.

    ```bash
    cd trulens
    pip install -e .
    ```
