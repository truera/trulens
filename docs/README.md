# How to build and view docs locally

Navigate to the TruLens root directory
```bash
cd trulens
```

Install docs requirements:
```bash
pip install -r docs/docs_requirements.txt
```

Build docs:
```bash
python -m mkdocs build --clean
python -m mkdocs build
```

Serve docs:
```bash
mkdocs serve
```
