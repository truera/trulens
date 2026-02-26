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

## TypeScript SDK

The TruLens TypeScript SDK lets you instrument Node.js applications with the
same OTEL-powered tracing used by the Python SDK. The packages are not yet
published to npm -- install from source.

**Prerequisites:** Node.js 18+ and [pnpm](https://pnpm.io/).

1. Clone the repo (if you haven't already).

    ```bash
    git clone https://github.com/truera/trulens.git
    cd trulens/typescript
    ```

2. Install dependencies and build the packages.

    ```bash
    pnpm install --no-frozen-lockfile
    pnpm --filter @trulens/semconv build
    pnpm --filter @trulens/core build
    ```

3. Build optional auto-instrumentation packages as needed.

    ```bash
    pnpm --filter @trulens/instrumentation-openai build   # OpenAI
    pnpm --filter @trulens/instrumentation-langchain build # LangChain.js
    ```

4. Link the packages into your project. From your app directory:

    ```bash
    pnpm add ../../trulens/typescript/packages/core
    pnpm add ../../trulens/typescript/packages/instrumentation-openai   # if needed
    pnpm add ../../trulens/typescript/packages/instrumentation-langchain # if needed
    ```

See the [TypeScript SDK guide](../component_guides/instrumentation/typescript.md) for usage details and the
[TypeScript Quickstart](quickstarts/typescript_quickstart.md) to get running in minutes.
