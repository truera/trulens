# ⛅ Optional Packages

Most of the examples included within `trulens_eval` require additional packages
not installed alongside `trulens_eval`. You may be prompted to install them
(with pip). The requirements file `trulens_eval/requirements.optional.txt`
contains the list of optional packages and their use if you'd like to install
them all in one go.

## Dev Notes

To handle optional packages and provide clearer instuctions to the user, we
employ a context-manager-based scheme (see `utils/imports.py`) to import
packages that may not be installed. The basic form of such imports can be seen
in `__init__.py`:

```python
with OptionalImports(messages=REQUIREMENT_LLAMA):
    from trulens.ext.instrument.llamaindex import TruLlama
```

This makes it so that `TruLlama` gets defined subsequently even if the import
fails (because `tru_llama` imports `llama_index` which may not be installed).
However, if the user imports TruLlama (via `__init__.py`) and tries to use it
(call it, look up attribute, etc), the will be presented a message telling them
that `llama-index` is optional and how to install it:

```
ModuleNotFoundError:
llama-index package is required for instrumenting llama_index apps.
You should be able to install it with pip:

    pip install "llama-index>=v0.9.14.post3"
```

If a user imports directly from TruLlama (not by way of `__init__.py`), they
will get that message immediately instead of upon use due to this line inside
`tru_llama.py`:

```python
OptionalImports(messages=REQUIREMENT_LLAMA).assert_installed(llama_index)
```

This checks that the optional import system did not return a replacement for
`llama_index` (under a context manager earlier in the file).

If used in conjunction, the optional imports context manager and
`assert_installed` check can be simplified by storing a reference to to the
`OptionalImports` instance which is returned by the context manager entrace:

```python
with OptionalImports(messages=REQUIREMENT_LLAMA) as opt:
    import llama_index
    ...

opt.assert_installed(llama_index)
```

`assert_installed` also returns the `OptionalImports` instance on success so
assertions can be chained:

```python
opt.assert_installed(package1).assert_installed(package2)
# or
opt.assert_installed[[package1, package2]]
```

### When to Fail

As per above implied, imports from a general package that does not imply an
optional package (like `from trulens ...`) should not produce the error
immediately but imports from packages that do imply the use of optional import
(`tru_llama.py`) should.
