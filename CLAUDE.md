# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Segment Studio (פילוח אוטומטי) is a student data-science project (Hebrew README) that discovers hidden customer segments in a CSV file via K-Means clustering, then uses an LLM to name and describe each cluster. The end goal per the assignment brief is a Streamlit app; development is currently happening in a single Jupyter notebook driven by `input()` prompts, with logic being ported to a plain script in parallel.

## Repository structure

- `Project segment studio.ipynb` — the main/working notebook: CSV load + normalization, K range selection, Elbow (WCSS) and Silhouette Score. This is the primary, actively-edited file; it is ahead of `project_main.py` in some places and behind it in others (e.g. the LLM-naming and CSV-export steps currently only exist in `project_main.py`).
- `project_main.py` — a script mirroring the notebook's pipeline end-to-end: load/normalize → Elbow/Silhouette over a user-chosen K range → final K-Means fit → per-cluster summary table → LLM naming/description via `ask_llm` → export to `<original_name>_clustered.csv`.
- `ask_llm.py` — a standalone duplicate of the LLM-calling loop in `project_main.py` (imports `df_summary` from it and re-runs the naming/description calls). Likely a scratch/experiment file rather than an imported module.
- `checklist.ipynb` / `checklist.html` — an interactive task checklist (ipywidgets-based) mirroring the assignment's stage-by-stage requirements; state is persisted to a local `checklist_state.json`. Not part of the app logic.
- `customers_clustered.csv` — sample output artifact from a prior run of the pipeline.
- README.md (Hebrew) documents the intended end-to-end flow and repo layout; treat it as the source of truth for planned scope (Streamlit app, bonuses) since the code hasn't caught up yet.

## Pipeline logic (applies to both the notebook and `project_main.py`)

1. Load a user-specified CSV path; keep a copy of the original (unmodified) DataFrame.
2. Drop any column whose name matches `id` (case-insensitive) and drop rows with missing values.
3. One-hot encode categoricals (`pd.get_dummies(..., drop_first=True)`) and standardize with `StandardScaler` — this normalized frame (`data_scaled`) is what K-Means trains on, never `df_original`.
4. Sweep a user-chosen K range, computing WCSS (`inertia_`) and Silhouette Score per K to help pick a final K.
5. Fit final K-Means at the chosen K; attach `cluster` labels back onto `df_original` (not the encoded frame).
6. Build a per-cluster summary (`df_summary`): numeric feature means per cluster, `count`, plus empty `name`/`description` columns to be filled by the LLM.
7. For each cluster row, call the LLM once for a short name and once for a short description, writing results into `df_summary`.
8. Export the clustered result as `<original_name>_clustered.csv`.

## LLM integration

`ask_llm()` (duplicated in `project_main.py` and `ask_llm.py`) POSTs to `https://ollama.com/api/chat` with model `gpt-oss:120b`, `stream: False`, using a Bearer token in the `Authorization` header.

**The API key is currently hardcoded in both files and committed to git.** When touching either file, replace it with an environment variable (e.g. `os.environ["OLLAMA_API_KEY"]`) and treat the exposed key as compromised — flag this to the user rather than silently perpetuating the pattern.

## Running

No build/lint/test tooling is configured yet (no `requirements.txt`, `pytest`, or linter config found). To run the current pipeline:

```
jupyter notebook "Project segment studio.ipynb"
```
or
```
python project_main.py
```

Both prompt interactively on stdin for the CSV path and the K range/final K — there is no non-interactive/CLI-argument mode yet.

Dependencies in use (no lockfile/requirements present): `numpy`, `pandas`, `seaborn`, `matplotlib`, `scikit-learn`, `requests`, and `ipywidgets` (checklist notebook only).
