# Virtual environments
Use uv to manage venvs and install packages

# Model definition
* Use pymc.dims as much as possible to help with model definitions and broadcasting
* When possible use non-centered parameterizations

# Test organization
* Tests are in `tests/` directory, organized by module (mirrors `chap_pymc/` structure)
* Test data in `tests/fixtures/data/`, config in `tests/fixtures/config/`
* Examples/CLI apps in `examples/` directory
* Run tests: `uv run pytest -v -m 'not slow'` (excludes slow tests by default)
* Run specific module: `uv run pytest tests/models/` or `pytest tests/curve_parametrizations/`
* Type checking only on production code: `uv run mypy chap_pymc/` (excludes tests/examples)

# Pre-push checks
Always make sure pytest passes before pushing code
Always make ruff checks pass before pushing code
Run mypy to verify type annotations: `uv run mypy chap_pymc/`
DONT access private variables in tests (e.g. _variable), use public API only