# Uninstalling TruLens

All TruLens packages are installed to the `trulens` namespace. Each package can be uninstalled with:

```bash
# Example
# pip uninstall trulens-core
pip uninstall trulens-<package_name>
```

To uninstall all TruLens packages, you can use the following command.

```bash
pip freeze | grep "trulens*" | xargs pip uninstall -y
```
