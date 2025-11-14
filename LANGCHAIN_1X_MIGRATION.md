# LangChain 1.x Migration Notes

## Dependency Conflict with NeMo Guardrails

### Problem

The `josh/langchain_one_support` branch adds support for `langchain>=1.0.0`, but this creates a dependency conflict:

- `trulens-apps-langgraph` requires `langchain>=1.0.0`
- `trulens-apps-nemo` depends on `nemoguardrails>=0.9`
- `nemoguardrails` requires `langchain>=0.2.14,<0.4.0` (incompatible with 1.x)

### Solution

**DEPRECATED and removed `trulens-apps-nemo` from langchain 1.x branches.** The package has been moved to an optional dependency group that is excluded from the lock file until `nemoguardrails` adds langchain 1.x support.

### Changes Made

#### 1. **`pyproject.toml`** (root)
- Removed `trulens-apps-nemo` from `[tool.poetry.group.apps.dependencies]`
- Created separate `[tool.poetry.group.nemo]` marked as `optional = true`
- Added deprecation comment

#### 2. **`src/apps/nemo/pyproject.toml`**
- Updated langchain constraint to match nemoguardrails: `langchain = ">=0.2.14,<0.4.0"`

#### 3. **`Makefile`**
- Updated `env-tests-optional` to use `--without nemo` flag
- Added explanatory comment

#### 4. **`src/apps/nemo/DEPRECATED.md`**
- Created deprecation notice explaining the incompatibility
- Documented workarounds for users who need NeMo Guardrails

#### 4. **`DEVELOPMENT.md`**
- Documented the incompatibility and install instructions

### Installation Instructions

**For langchain 1.x development (current branch):**
```bash
# Nemo is automatically excluded (it's in an optional group)
poetry install --with dev --with apps
poetry lock  # Now works without conflicts
```

**For nemo development (requires langchain <1.0):**
```bash
# Switch to a pre-langchain-1.x branch first
git checkout <pre-1.x-branch>
poetry install --with dev --with nemo
```

**For users (when packages are published):**
```bash
# Default: includes langchain 1.x support, excludes nemo
pip install trulens[langchain,langgraph,llamaindex]

# For nemo (incompatible with langgraph on langchain 1.x):
pip install trulens[langchain,nemo]  # Will use langchain<1.0
```

### CI/CD Impact

**Jenkinsfile** ✅ Already updated
- Uses `make test-e2e-stable` and `make test-notebook-stable`
- These call `env-tests-optional` which now excludes nemo
- No changes needed to Jenkinsfile

**Local Development** ✅ Documented
- Developers working on langchain 1.x: `poetry install --without nemo`
- Developers working on nemo: work on a pre-1.x branch

### When Can We Re-enable Nemo?

Monitor `nemoguardrails` releases: https://pypi.org/project/nemoguardrails/

When they release a version supporting `langchain>=1.0.0`:
1. Update `src/apps/nemo/pyproject.toml` langchain constraint
2. Move `trulens-apps-nemo` back to `[tool.poetry.group.apps.dependencies]` in root `pyproject.toml`
3. Update `Makefile` `env-tests-optional` to include nemo again
4. Run `poetry lock` to regenerate lockfile

### Testing

**Run all tests (excluding nemo):**
```bash
poetry install --with dev --with apps --without nemo
TEST_OPTIONAL=1 make test-unit
```

**Verify OTEL tests with isolation:**
```bash
poetry install --with dev --with apps --without nemo
TEST_OPTIONAL=1 make test-unit  # Uses pytest-xdist for OTEL test isolation
```
