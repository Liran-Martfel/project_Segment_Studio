# Segment Studio — Streamlit UI & Integration Build Prompt

## Context
This is a course project ("Segment Studio"): a Streamlit app that lets a user
upload a CSV, run K-Means clustering, visualize the elbow/WCSS curve, generate
human-readable cluster names/descriptions via an LLM (Ollama, local), and
export the labeled data back to CSV.

**Important: the core ML/plotting logic already exists in this repo** —
data loading, preprocessing, K-Means, WCSS/elbow calculation, and
matplotlib/seaborn plotting functions are already written and working
standalone (not yet wired into Streamlit).

Your job is NOT to rewrite the ML logic. Your job is to:
1. Audit the existing code first (list the functions/modules already present
   and what each one does) before writing anything.
2. Build the missing Streamlit UI/orchestration layer that calls into that
   existing logic.
3. Add the Ollama LLM integration for cluster naming.
4. Add the bonus features.
5. Prepare the repo for GitHub submission.

Do not duplicate logic that already exists — import and call it. If you find
an existing function doesn't cleanly support what the UI needs (e.g. it plots
directly instead of returning a figure), refactor that function minimally
rather than writing a parallel version.

---

## Step 0 — Audit (do this first, report back before coding)
- List every existing .py file and its functions/classes.
- Identify: CSV loading, preprocessing/normalization, K-Means runner,
  WCSS/elbow calculator, any existing silhouette score code, plotting
  functions, and any existing Ollama call code.
- Flag anything that needs refactoring to be UI-friendly (e.g., a function
  that does `plt.show()` instead of returning a `Figure`).

---

## App structure (Streamlit, single multi-step page or sidebar-nav — your call,
but keep state in `st.session_state` so results persist across steps)

### Step 1 — Upload & preview
- File uploader restricted to `.csv`, max ~200MB.
- Load into a pandas DataFrame using the existing loader function.
- Display the dataframe (`st.dataframe`).
- Store the raw df and the uploaded filename (without extension) in
  `st.session_state` — the filename is needed later for the export naming
  convention.

### Step 2 — Elbow / WCSS
- Two sliders: `Min k` and `Max k` (bounds e.g. 2–20).
- "Run WCSS" button.
- For each k in range, run K-Means (reuse existing function) and collect
  WCSS. Also compute **Silhouette Score** per k (bonus — see below) using
  the existing preprocessing pipeline.
- Show results table: columns `k`, `WCSS`, `Silhouette` (only Silhouette if
  bonus is enabled — see Bonus section).
- Plot the Elbow curve (k on X, WCSS on Y) using the existing plotting code,
  adapted to return a `matplotlib.figure.Figure` so Streamlit can render it
  with `st.pyplot(fig)`.

### Step 3 — Choose K and create clusters
- Slider "Select k".
- **Auto-K button (bonus)** — see Bonus section for logic.
- "Create clusters" button: run K-Means with the chosen k, assign
  `cluster_id` to every row, and build a summary table:
  `cluster_id`, `count`, `name` (empty), `description` (empty).
- Store the clustered df and the cluster summary table in session_state.

### Step 4 — LLM cluster naming (Ollama)
- For each cluster, build a compact stats summary:
  - number of observations in the cluster
  - mean of each numeric feature
  - most frequent value of each categorical feature (if any)
- Send that summary to a local Ollama model and ask it to return, per
  cluster:
  - a short cluster name
  - a one-sentence description
- **Prompt the LLM to return strict JSON** (e.g.
  `{"name": "...", "description": "..."}`) so the response can be parsed
  reliably — include a fallback parser (regex/try-except) in case the model
  wraps it in markdown or extra text.
- Wire this to an Ollama client call (assume `ollama` Python package or
  local HTTP API at `http://localhost:11434`). Model name should be a
  constant/config variable, not hardcoded inline, e.g.
  `OLLAMA_MODEL = "llama3"` — I'll adjust to whichever model I actually have
  pulled.
- Button: "Generate names/descriptions with Ollama".
- Update the cluster summary table in place with the returned `name` and
  `description` columns.
- Handle errors gracefully (Ollama not running, timeout, malformed JSON) —
  show a Streamlit warning/error, don't crash the app.

### Step 5 — Export
- "Download clustered CSV" button.
- Take the **original** CSV, add a new column `cluster_name` mapping each
  row's cluster_id to its LLM-generated name.
- File name convention: `<original_filename>_clustered.csv`
  (use the filename stored in session_state from Step 1).
- Use `st.download_button` with the CSV bytes generated via
  `df.to_csv(index=False).encode("utf-8")`.

---

## Bonus features (include all)

1. **Silhouette Score instead of/alongside WCSS**
   - Compute silhouette score for each k in the Step 2 range.
   - Show both metrics in the results table so the user can compare.

2. **Auto-K selection button** (in Step 3)
   - If silhouette scores are available: auto-select the k with the
     **highest** silhouette score.
   - If only WCSS is available: send the WCSS table to the Ollama LLM and
     ask it to recommend the best k (e.g. via elbow heuristic), parse the
     numeric answer, and pre-fill the slider.
   - Show the reasoning/result to the user (e.g. "Auto-selected k=5 based on
     highest silhouette score: 0.62").

3. **Outlier cleaning**
   - After clustering, compute each point's distance from its assigned
     cluster centroid.
   - Flag points whose distance is an outlier (e.g. beyond N standard
     deviations from the mean distance within their cluster, or using IQR).
   - Add a toggle/checkbox: "Remove outliers before export" — if enabled,
     drop flagged rows from the final clustered/exported dataframe and show
     how many rows were removed.

---

## GitHub submission prep
- Ensure repo has:
  - `requirements.txt` (pin versions actually used: streamlit, pandas,
    scikit-learn, matplotlib, seaborn, ollama or requests, etc.)
  - `README.md` covering: project description, setup instructions
    (including "make sure Ollama is running locally and you've pulled the
    model with `ollama pull <model>`"), how to run
    (`streamlit run app.py`), and a short description of each step/feature.
  - Clean folder structure (e.g. `app.py` or `main.py` at root, a `src/` or
    `utils/` folder for the existing ML logic, no stray notebook/scratch
    files committed).
  - `.gitignore` for `venv/`, `__pycache__/`, `.streamlit/secrets.toml` if
    used, uploaded CSVs, etc.
- Do not commit any sample data files unless they're meant as demo/test
  fixtures — ask me if unsure which files are demo vs. scratch.

---

## Style/constraints
- Keep the existing ML/plotting code as the single source of truth; the UI
  layer should be a thin wrapper calling into it.
- Use `st.session_state` for all cross-step data (raw df, clustered df,
  WCSS/silhouette table, cluster summary table, original filename) so
  navigating between steps doesn't lose progress.
- Add minimal inline comments explaining non-obvious logic (esp. the
  outlier detection and auto-K logic), since this is a course submission.
- After building, run the app locally (or at least a syntax/import check)
  and report back what you verified vs. what still needs manual testing
  (e.g. actual Ollama calls, since that depends on my local model).
