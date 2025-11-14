# DEPRECATED: trulens-apps-nemo

## Status: Removed from langchain 1.x branches

The `trulens-apps-nemo` package has been temporarily removed from branches that support `langchain>=1.0.0` due to a dependency conflict.

## Reason

- `trulens-apps-nemo` depends on `nemoguardrails>=0.9`
- `nemoguardrails` requires `langchain>=0.2.14,<0.4.0`
- This is incompatible with `trulens-apps-langgraph` which requires `langchain>=1.0.0`

## Impact

- The package is **removed from the root `pyproject.toml`** on langchain 1.x branches
- Users cannot install both `trulens-apps-langgraph` and `trulens-apps-nemo` in the same environment
- NeMo Guardrails integration is unavailable when using langchain 1.x

## Future

This package will be re-enabled when one of the following occurs:

1. **`nemoguardrails` adds langchain 1.x support** - Monitor: https://pypi.org/project/nemoguardrails/
2. **We create a compatibility layer** that works with both langchain versions
3. **We maintain separate branches** for langchain <1.0 and >=1.0

## Workaround for Users

If you need NeMo Guardrails support:

1. **Use a langchain <1.0 environment:**
   ```bash
   pip install "langchain<1.0" "trulens-apps-nemo>=2.4.2"
   ```

2. **Do not install `trulens-apps-langgraph` in the same environment**

3. **Use pre-langchain-1.x TruLens versions** if you need both langgraph and nemo (not recommended)

## For Developers

**Working on langchain 1.x features:**
```bash
poetry install --with dev --with apps --without nemo
```

**Working on nemo features (requires langchain <1.0):**
- Check out a pre-1.x branch
- Or manually adjust langchain version constraints

## Questions

Contact the TruLens team or check:
- GitHub Issues: https://github.com/truera/trulens/issues
- Documentation: https://trulens.org/
