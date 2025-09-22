# Repository Guidelines

## Project Structure & Module Organization
- `src/` hosts the Python package: `data_generation/` (query pipelines), `evals/` (dietary checks), helpers like `constants.py`.
- `database/` contains Alembic migrations, reusable utilities, and CLI scripts; run them with the repo root on `PYTHONPATH`.
- `data/`, `output/`, and `prompts/` track raw inputs, generated artifacts, and prompt templates. Commit sanitized samples only and keep heavy files ignored.
- Use `notebooks/` for experiments; persist derived assets in `output/` rather than notebook cells.

## Build, Test, and Development Commands
- Bootstrap the environment with `uv sync`, then run tools via `uv run <command>`.
- `make load-consumables INPUT=<csv>` streams rows into Postgres; add `THRESHOLD`, `BATCH_SIZE`, or `DRY_RUN` as needed.
- `make generate-queries ESCI=<E|S|C|I>` fronts `src/data_generation/initial_generation.py`; extend with `LIMIT`, `OUTPUT_PATH`, or `MODEL`.
- For ad-hoc runs, prefer `uv run python database/test_connection.py` and `uv run python src/data_generation/initial_generation.py --help`.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, snake_case for functions and files, PascalCase for classes, and ALL_CAPS constants.
- Type hints plus succinct docstrings are expected on public functions; mirror the structured logging setup used in `initial_generation.py`.
- Inject configuration through `.env` (loaded with `dotenv`) and never hardcode secrets or paths.

## Testing Guidelines
- No automated suite exists yet; when you add meaningful logic, create `pytest` modules under `tests/` (install with `uv add pytest` if needed).
- Validate database flows with `uv run python database/test_connection.py` against staging credentials, and honour `--confirm` flags before destructive actions.
- Attach sample outputs in `output/` (gitignored) and describe manual checks in pull requests.

## Commit & Pull Request Guidelines
- Match the concise, imperative commit style in `git log` (e.g., “add dietary evals”); keep each commit scoped.
- Reference issues or tickets upfront, and document data dependencies, migrations, or prompt revisions in the body.
- PRs should outline intent, reproduction commands, config expectations (`.env` keys, connection details), and screenshots or sample rows when behavior shifts.

## Data & Configuration Tips
- Keep `.env` local, set `root_folder` so scripts resolve imports, and share new keys in tooling docs.
- Note any external datasets in `context/` or the PR, and provide anonymized subsets under `data/` for reviewers.
- Rotate `query_generation.log` and `load_consumables.log` before opening a PR to avoid noisy diffs.
