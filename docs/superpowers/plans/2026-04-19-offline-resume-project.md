# Offline-First Gamma Exposure Resume Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current repository into a clean, interview-defensible quantitative finance resume project whose main artifact is an offline teaching notebook and whose runtime path depends only on local Parquet files under `data/raw/`.

**Architecture:** Keep the existing quant research core (`src/data`, `src/exposure`, `src/intraday`, `src/research`) and remove presentation surfaces that do not support the resume goal. Split the data layer into two explicit paths: an optional one-time ClickHouse refresh path that writes canonical raw Parquet files, and a required offline analysis path that reads only those raw files, builds derived datasets, writes simple non-HTML artifacts, and powers the teaching notebook.

**Tech Stack:** Python 3.12, Polars, scikit-learn, scipy, typer, pytest, Jupyter notebook execution tooling

---

## Repo Diagnosis

- The repository already has a strong, explainable research core in `src/exposure/`, `src/intraday/`, and `src/research/`.
- The current product surface is misaligned with the new goal. `src/app/streamlit_app.py`, `src/reporting/`, committed HTML reports in `outputs/`, and `plotly` / `streamlit` dependencies make the repo feel like a mini app rather than a research project.
- Offline portability is only partial today. The code can reuse Parquet cache files, but the canonical raw files live in `data/cache/`, the notebook still calls cache-first fetchers that can fall back to ClickHouse, and `.env` / database behavior still leaks into the normal runtime story.
- The existing notebook is a useful start, but it is in `notebooks/project_demo/` instead of directly under `notebooks/`, it does not strictly alternate one substantial Markdown cell then one code cell all the way through, and it still uses data-loading functions that preserve live-database behavior.
- Current raw data coverage is small but real: local Parquet files cover `2024-01-02` through `2024-01-31`, with about `4.3 MB` total raw payload. That is enough to prove the pipeline works, but it is thin for regime analysis, leave-one-month-out checks, and predictive evaluation. A one-time refresh to a longer 2024 window is justified if ClickHouse access is available.

## Target File Map

### Keep and Refocus

- `src/data/clickhouse_client.py`
- `src/exposure/aggregation.py`
- `src/exposure/cleaning.py`
- `src/intraday/metrics.py`
- `src/research/bootstrap.py`
- `src/research/dataset.py`
- `src/research/descriptive.py`
- `src/research/multi_factor.py`
- `src/research/predictive.py`
- `src/research/regime.py`
- `src/research/statistical_tests.py`

### Likely New Files

- `src/data/raw_store.py`
- `src/data/raw_cache_builder.py`
- `src/pipeline/__init__.py`
- `src/pipeline/offline_pipeline.py`
- `src/GUIDE_src.md`
- `src/data/GUIDE_data.md`
- `src/exposure/GUIDE_exposure.md`
- `src/intraday/GUIDE_intraday.md`
- `src/research/GUIDE_research.md`
- `src/pipeline/GUIDE_pipeline.md`
- `docs/reference/offline-data-contract.md`
- `docs/reference/clickhouse-raw-cache-refresh.md`
- `notebooks/gamma_exposure_pipeline_demo.ipynb`
- `tests/data/test_raw_store.py`
- `tests/data/test_raw_cache_builder.py`
- `tests/pipeline/test_offline_pipeline.py`

### Likely Modified Files

- `.gitignore`
- `README.md`
- `GUIDE_ROOT.md`
- `pyproject.toml`
- `uv.lock`
- `src/cli.py`
- `src/config.toml`
- `src/settings.py`
- `tests/test_cli_smoke.py`
- `tests/test_settings.py`
- `tests/data/test_queries.py`

### Likely Deleted Files and Folders

- `src/app/__init__.py`
- `src/app/streamlit_app.py`
- `src/reporting/__init__.py`
- `src/reporting/charts.py`
- `src/reporting/html_report.py`
- `tests/reporting/test_html_report.py`
- `notebooks/project_demo/gamma_exposure_engine_demo.ipynb`
- `outputs/SPY_2024-01-02_2024-01-03_gamma_report.html`
- `outputs/SPY_2024-01-02_2024-01-10_gamma_report.html`
- `outputs/SPY_2024-01-02_2024-01-31_gamma_report.html`
- `outputs/samples/SPY_2024-01-02_2024-01-31_gamma_report.html`
- `dist/`
- `src/gamma_exposure_engine.egg-info/`

---

### Task 1: Define the offline-first runtime contract and clean repository boundaries

**Files:**
- Modify: `.gitignore`
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `src/config.toml`
- Modify: `src/settings.py`
- Create: `docs/reference/offline-data-contract.md`
- Create: `src/GUIDE_src.md`

- [ ] Remove presentation-first dependencies and references from the public project story. Drop `plotly`, `streamlit`, and `jinja2` from `pyproject.toml` if they are no longer needed anywhere after the refactor. Keep `clickhouse-connect`, `polars`, `scikit-learn`, `scipy`, and `typer`.
- [ ] Rewrite the runtime contract in `README.md` and `GUIDE_ROOT.md` so the default workflow is: clone repo, ensure `data/raw/` contains the shipped Parquet files, run offline pipeline, open notebook. Treat ClickHouse as an optional refresh path only.
- [ ] Change config and settings names away from `cache_dir = "data/cache"` semantics. The normal path should resolve `data/raw`, while any optional refresh helper should treat that directory as the destination for canonical raw files.
- [ ] Update `.gitignore` so committed raw data lives only under `data/raw/`. Stop tracking `data/cache/` and stop keeping committed HTML files under `outputs/`.
- [ ] Add `docs/reference/offline-data-contract.md` documenting the canonical raw inputs:
  - `data/raw/SPY_intraday_bars.parquet`
  - `data/raw/SPY_options_snapshot.parquet`
  - required columns and basic meaning
  - expected date coverage
  - promise that offline analysis never reaches for ClickHouse

### Task 2: Consolidate raw data into `data/raw` and make the demo payload portable

**Files:**
- Delete: `data/cache/SPY_intraday_bars.parquet`
- Delete: `data/cache/SPY_intraday_bars.metadata.json`
- Delete: `data/cache/SPY_options_snapshot.parquet`
- Delete: `data/cache/SPY_options_snapshot.metadata.json`
- Create or Modify: `data/raw/SPY_intraday_bars.parquet`
- Create or Modify: `data/raw/SPY_options_snapshot.parquet`
- Create: `data/raw/manifest.json`
- Create: `docs/reference/clickhouse-raw-cache-refresh.md`

- [ ] Promote `data/raw/` to the only raw-data location in the repository. There should be no second copy under `data/cache/`, `outputs/`, notebooks, or any other folder.
- [ ] Decide the shipped demo window before writing code:
  - preferred: full calendar year `2024-01-02` through `2024-12-31` if the combined Parquet payload stays comfortably under `100 MB`
  - acceptable fallback: six months of 2024 if one year unexpectedly exceeds the target
  - last resort: keep the existing January 2024 sample only if ClickHouse is unavailable during execution
- [ ] Use ClickHouse once, only if needed, to refresh or expand the raw cache. Write the final canonical files directly into `data/raw/`, not `data/cache/`.
- [ ] Write `data/raw/manifest.json` with:
  - dataset names
  - symbol
  - date range
  - row counts
  - file sizes
  - schema version
  - a short note that these files are the offline demo source of truth
- [ ] Keep packaging simple unless the data size forces complexity. Prefer two canonical Parquet files over monthly parts. Only split by month or quarter if one file becomes unwieldy or total size materially threatens the `100 MB` target.
- [ ] Document the refresh procedure in `docs/reference/clickhouse-raw-cache-refresh.md`, but make clear that it is a one-time maintenance workflow, not part of normal execution or notebook use.

### Task 3: Separate optional ClickHouse refresh from mandatory offline analysis

**Files:**
- Create: `src/data/raw_store.py`
- Create: `src/data/raw_cache_builder.py`
- Modify: `src/data/clickhouse_client.py`
- Modify: `src/data/intraday_queries.py`
- Modify: `src/data/options_queries.py`
- Modify: `src/cli.py`
- Modify: `src/settings.py`
- Create: `tests/data/test_raw_store.py`
- Create: `tests/data/test_raw_cache_builder.py`
- Modify: `tests/data/test_queries.py`

- [ ] Add `src/data/raw_store.py` with explicit local-only loaders such as `load_raw_intraday_bars()` and `load_raw_options_snapshot()`. These functions must:
  - read only from `data/raw`
  - fail with actionable errors if files are missing
  - never invoke ClickHouse
  - never depend on `.env`
- [ ] Add `src/data/raw_cache_builder.py` as the only module allowed to use ClickHouse for cache population. Its responsibility is one-time raw-data refresh into `data/raw`.
- [ ] Narrow the role of `src/data/intraday_queries.py` and `src/data/options_queries.py`. They should either become pure ClickHouse extractors used only by the refresh path, or be absorbed into `raw_cache_builder.py` if that simplifies the design.
- [ ] Replace the current `report`-oriented CLI in `src/cli.py` with two explicit commands:
  - one optional refresh command for ClickHouse to raw Parquet
  - one required offline analysis command that reads from local Parquet only
- [ ] Make the offline analysis command write simple non-frontend artifacts under `outputs/`, for example:
  - aligned research dataset as Parquet
  - quantile summary as CSV
  - statistical test summary as CSV
  - regime summary as CSV
  - predictive comparison as CSV
  - a small run manifest as JSON
- [ ] Update tests so they verify:
  - raw loaders work with only local Parquet
  - offline analysis fails clearly when raw files are absent
  - refresh code is the only place that touches ClickHouse

### Task 4: Remove all frontend and HTML/report-generation concerns

**Files:**
- Delete: `src/app/__init__.py`
- Delete: `src/app/streamlit_app.py`
- Delete: `src/reporting/__init__.py`
- Delete: `src/reporting/charts.py`
- Delete: `src/reporting/html_report.py`
- Delete: `tests/reporting/test_html_report.py`
- Modify: `tests/test_cli_smoke.py`
- Delete: `outputs/*.html`
- Delete: `outputs/samples/`

- [ ] Delete the Streamlit layer entirely. The project does not need an explorer or dashboard surface.
- [ ] Delete the HTML reporting layer entirely. The project does not need a browser artifact, HTML templating, or dashboard-like rendering logic.
- [ ] Rewrite `tests/test_cli_smoke.py` around the new offline CLI contract. Replace HTML assertions with assertions about written Parquet, CSV, and JSON artifacts.
- [ ] Remove all README, guide, and notebook references to Streamlit, Plotly, HTML reports, or browser-openable sample outputs.
- [ ] Clean the repository of generated build artifacts and stale output artifacts:
  - `dist/`
  - `src/gamma_exposure_engine.egg-info/`
  - committed HTML outputs

### Task 5: Build the final offline pipeline module and make the notebook the teaching centerpiece

**Files:**
- Create: `src/pipeline/__init__.py`
- Create: `src/pipeline/offline_pipeline.py`
- Create: `src/pipeline/GUIDE_pipeline.md`
- Create: `notebooks/gamma_exposure_pipeline_demo.ipynb`
- Delete: `notebooks/project_demo/gamma_exposure_engine_demo.ipynb`
- Create: `tests/pipeline/test_offline_pipeline.py`

- [ ] Add `src/pipeline/offline_pipeline.py` to orchestrate the full offline workflow:
  - load raw intraday and options Parquet
  - build spot close
  - enrich options with spot
  - clean options
  - build daily gamma factors
  - build daily realized-variance and response metrics
  - build aligned research dataset
  - run descriptive, inferential, regime, robustness, and predictive modules
  - write non-HTML artifacts
- [ ] Keep the pipeline explainable and linear. Do not add a framework, task runner, or abstraction layer that makes the story harder to explain in an interview.
- [ ] Replace the current notebook with `notebooks/gamma_exposure_pipeline_demo.ipynb` directly under `notebooks/`.
- [ ] Enforce the notebook cell pattern from start to finish:
  - one substantial Markdown cell
  - one code cell
  - repeat
- [ ] Make every Markdown cell teach the next code cell in plain language. It must define symbols, explain the theory, name the inputs and outputs, and say exactly what the next code cell is about to produce.
- [ ] Make the notebook explicitly offline:
  - start by naming the two raw Parquet files
  - explain the raw-cache contract briefly
  - state clearly that the notebook does not connect to ClickHouse
  - use `raw_store.py` loaders, not cache-first fetchers
- [ ] Cover the full teaching pipeline inside the notebook except live database pulling:
  - raw data inspection
  - spot-close construction
  - cleaning diagnostics
  - gamma factor construction
  - realized variance and response metrics
  - dataset alignment
  - descriptive summary
  - statistical tests
  - regime analysis
  - robustness checks
  - predictive evaluation
  - interpretation of results and limitations

### Task 6: Add navigation guides and documentation that support interview explanation and a later blog post

**Files:**
- Modify: `README.md`
- Modify: `GUIDE_ROOT.md`
- Create: `src/data/GUIDE_data.md`
- Create: `src/exposure/GUIDE_exposure.md`
- Create: `src/intraday/GUIDE_intraday.md`
- Create: `src/research/GUIDE_research.md`
- Create: `docs/reference/offline-data-contract.md`
- Create: `docs/reference/clickhouse-raw-cache-refresh.md`

- [ ] Rewrite `README.md` around the new hiring narrative:
  - what question the project studies
  - what raw files are needed
  - how to run offline
  - how to refresh raw files optionally
  - what the notebook teaches
  - how to explain the project in an interview
- [ ] Update `GUIDE_ROOT.md` so it matches the new architecture exactly and stops describing HTML reporting or Streamlit.
- [ ] Add missing `GUIDE_*.md` files under `src/` so the folder structure is self-explanatory for future agents and for you.
- [ ] Document the blog-ready backbone explicitly:
  - which intermediate artifacts matter
  - where the main tables come from
  - how the notebook can be converted into a blog outline later
- [ ] Document limitations honestly:
  - empirical association, not causal inference
  - sample window dependence
  - single-symbol scope
  - sensitivity to options data conventions and cleaning rules

### Task 7: Full execution verification, notebook execution, and cleanup

**Files:**
- Modify: `tests/test_cli_smoke.py`
- Modify: `tests/test_settings.py`
- Create or Modify: notebook execution verification tooling if needed

- [ ] Run unit and integration tests after the refactor:
  - `uv run pytest -v`
- [ ] Run the offline CLI against the committed raw Parquet files:
  - example command: `uv run gex run-offline-analysis --start 2024-01-02 --end 2024-12-31 --output-dir outputs/demo`
  - if the shipped sample is shorter, use the exact manifest range instead
- [ ] Verify the output directory contains the expected non-HTML artifacts and that the key tables are non-empty.
- [ ] Execute the final notebook from top to bottom before delivery. Use a reproducible command rather than a claim:
  - preferred: `uv run jupyter nbconvert --to notebook --execute notebooks/gamma_exposure_pipeline_demo.ipynb --output gamma_exposure_pipeline_demo.executed.ipynb`
  - acceptable alternative: use `nbclient` if that is the lighter dependency choice
- [ ] Confirm the notebook executes without ClickHouse access by running it against the committed Parquet files only.
- [ ] Re-run `uv run pytest -v` after any notebook-driven fixes.
- [ ] Finish with repository hygiene:
  - remove stale output artifacts not meant to be committed
  - `git add .`
  - `git commit -m "refactor: make gamma exposure project offline-first and notebook-led"`

---

## Dependency Order

1. Task 1 first: the offline contract and data location must be locked before moving code.
2. Task 2 second: the canonical raw files and their size budget determine every later loader, notebook path, and README instruction.
3. Task 3 before Task 4: replace the analysis runtime before deleting the current app/report surface.
4. Task 4 before Task 5: once frontend and HTML concerns are gone, the notebook and pipeline can teach the final architecture instead of a transitional one.
5. Task 6 after the refactor shape is stable: docs should describe the final structure, not the migration path.
6. Task 7 last: full verification only matters after the final code, notebook, data layout, and docs are all in place.

## Main Risks

- The current January 2024 sample may be too short for a convincing research narrative. If ClickHouse refresh is impossible, the project should downscope claims and keep the notebook honest about sample-size limits.
- A careless refactor could leave hidden ClickHouse fallbacks in the notebook path. The implementation agent must treat any fallback logic in the offline path as a bug.
- Deleting reporting code too early can leave the CLI without a success path. Replace the runtime surface first, then delete the old surface.
- Moving raw files without updating tests and `.gitignore` can create duplicate data and a messy repository state.
- Notebook execution can fail if it depends on implicit state, missing kernels, or a date range longer than the shipped raw data supports. The execution pass must use the exact committed manifest range.

## Verification Checklist

- `data/raw/` is the only committed raw-data location.
- No HTML, Streamlit, Plotly, or dashboard code remains anywhere in the active runtime path.
- The offline CLI runs successfully with the committed Parquet files and no ClickHouse access.
- The notebook lives directly under `notebooks/`, executes top to bottom, and alternates one Markdown cell then one code cell throughout.
- The notebook explains the full pipeline from local raw Parquet through interpretation.
- `README.md`, `GUIDE_ROOT.md`, and the new `GUIDE_*.md` files all describe the same final architecture.
- `uv run pytest -v` passes after the final refactor.

## Offline Portability Guarantee

- Canonical raw inputs are committed under `data/raw/` and nowhere else.
- The default runtime path uses only `src/data/raw_store.py` loaders that read local Parquet and fail fast when files are absent.
- ClickHouse code lives behind a separate refresh command and separate documentation, so it is not part of normal execution.
- The notebook imports only offline loaders and offline pipeline functions.
- The README quick-start uses no `.env`, no database, and no browser surface.

## Demo Data Packaging Recommendation

- Recommended default: ship two canonical files:
  - `data/raw/SPY_intraday_bars.parquet`
  - `data/raw/SPY_options_snapshot.parquet`
- Recommended target coverage: as much of calendar year 2024 as fits comfortably under the raw payload budget.
- Recommended size guardrail: keep combined raw Parquet under `100 MB`, and prefer staying below roughly `60 MB` if the full-year data allows it.
- Recommended packaging fallback: split by month only if one-file packaging becomes materially awkward; otherwise keep the loader and notebook simple with one file per dataset.
- Always record the final row counts, date range, and sizes in `data/raw/manifest.json`.

## Recommended Scope Cut

- Keep the project single-symbol and offline-first. Do not add new symbols, new markets, or new model families.
- Do not repackage the whole codebase into a deeper namespace unless it becomes necessary for the refactor. The repo already has meaningful domain folders under `src/`.
- Do not add a frontend, HTML report, dashboard, web app, or static site export.
- Do not add new research features unless a small fix is needed to make the current modules more explainable or runnable offline.
- Treat the notebook as the main human-facing artifact and the offline CLI as the reproducible execution path. Everything else is supporting infrastructure.

