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

# Commits
* Always commit after completing a logical unit of work with passing tests
* Always tag commits with jira ids

# Jira
* epics are large projects/pieces of work that can take up to 6 months to finish
* features are isolated pieces of work that takes typically much less time (pieces of an epic).
* always suggest an update to jira after pushing to master
* Be succinct and not verbose when creating/updating jira work items/epics/

## Project Context

**Project**: Climate (CLIM)
**Organization**: DHIS2
**Main Epic**: CLIM-140 - chap_pymc model
- Goal: Publish a monthly version of the curve parametrization based pymc model

### Active Tasks

**CLIM-141**: Implement proper NormalMixture model for seasonal patterns
- Replace current hacky weighted average with PyMC's NormalMixture
- Two components: (1) Empty season baseline, (2) Normal seasonal Fourier signal
- Each observation comes from ONE component (not weighted average)
- Proper statistical mixture model for climate-driven disease dynamics
- Related code: `fourier_parametrization.py:38-54` (_mixture_weights method)
- Status: To be implemented