"""Segment Studio — Streamlit UI wrapping the K-Means/LLM clustering pipeline."""

import os

import pandas as pd
import streamlit as st

from src import llm, pipeline

st.set_page_config(page_title="Segment Studio", page_icon="🧩", layout="wide")

# On a cloud deployment there's no local Ollama to reach, so pull an
# OLLAMA_API_KEY from Streamlit secrets (set in the app's Secrets panel on
# share.streamlit.io, never committed to git) into the environment, where
# src/llm.py picks it up and switches to Ollama Cloud automatically.
try:
    if "OLLAMA_API_KEY" in st.secrets:
        os.environ.setdefault("OLLAMA_API_KEY", st.secrets["OLLAMA_API_KEY"])
except Exception:
    pass  # no secrets.toml locally -- fine, local Ollama is used instead

st.markdown(
    """
    <style>
    h1, h2, h3 { color: #7C3AED; }
    div.stButton > button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

STEP_LABELS = [
    "1 · Upload",
    "2 · Find k",
    "3 · Clusters",
    "4 · Name groups",
    "5 · Export",
]

if "step" not in st.session_state:
    st.session_state.step = 1
if "max_step_reached" not in st.session_state:
    st.session_state.max_step_reached = 1


def go_to(step: int) -> None:
    st.session_state.step = step
    st.session_state.max_step_reached = max(st.session_state.get("max_step_reached", 1), step)
    st.rerun()


def render_stepper(current: int) -> None:
    """Clickable step pills — jump freely between any step already reached."""
    cols = st.columns(len(STEP_LABELS))
    for i, (col, label) in enumerate(zip(cols, STEP_LABELS), start=1):
        unlocked = i <= st.session_state.max_step_reached
        if col.button(
            label, key=f"stepper_{i}",
            disabled=not unlocked,
            type="primary" if i == current else "secondary",
            width="stretch",
        ):
            go_to(i)


def display_cluster_table(df: pd.DataFrame, show_name_description: bool = True) -> None:
    """Render the cluster summary with narrow numeric columns and wide name/description."""
    if not show_name_description:
        df = df.drop(columns=["name", "description"], errors="ignore")
    column_config = {}
    for col in df.columns:
        if col in ("name", "description"):
            column_config[col] = st.column_config.TextColumn(col, width="large")
        else:
            column_config[col] = st.column_config.Column(col, width="small")
    st.dataframe(df, column_config=column_config, width="stretch", hide_index=True)


st.title("🧩 Segment Studio")
st.caption("Upload a CSV, discover hidden customer segments with K-Means, and let an LLM name them.")

render_stepper(st.session_state.step)

step = st.session_state.step

# ---------------------------------------------------------------------------
# Step 1 — Upload & preview
# ---------------------------------------------------------------------------
if step == 1:
    st.header("Upload your data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        df_encoded, df_original, data_scaled = pipeline.load_and_normalize(raw_df)

        st.session_state["df_original"] = df_original
        st.session_state["data_scaled"] = data_scaled
        st.session_state["uploaded_name"] = os.path.splitext(uploaded_file.name)[0]

        st.dataframe(df_original, width="stretch")

    ready = "df_original" in st.session_state
    if not ready:
        st.info("Upload a CSV to continue.")
    if st.button("Continue", disabled=not ready, type="primary"):
        go_to(2)

# ---------------------------------------------------------------------------
# Step 2 — Elbow / WCSS
# ---------------------------------------------------------------------------
elif step == 2:
    st.header("Find the right number of groups")

    n_samples = len(st.session_state["data_scaled"])
    slider_max = max(2, min(20, n_samples - 1))

    col1, col2 = st.columns(2)
    k_min = col1.slider("Min k", min_value=2, max_value=slider_max, value=2)
    k_max = col2.slider("Max k", min_value=2, max_value=slider_max, value=min(10, slider_max))

    if st.button("Run WCSS"):
        if k_max <= k_min:
            st.error("Max k must be greater than Min k.")
        else:
            with st.spinner(f"Running K-Means for k={k_min}..{k_max}..."):
                st.session_state["results_df"] = pipeline.compute_wcss_silhouette(
                    st.session_state["data_scaled"], k_min, k_max
                )

    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        best_k, _ = pipeline.auto_select_k_by_silhouette(results_df)
        best_score = results_df.loc[results_df["k"] == best_k, "silhouette"].iloc[0]

        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Best k (by silhouette)", best_k)
        metric_col2.metric("Silhouette score", f"{best_score:.3f}")

        chart_col, _spacer = st.columns([2, 1])
        chart_col.pyplot(pipeline.plot_elbow(results_df, best_k=best_k), width="content")

        st.caption("WCSS / Silhouette table")
        st.dataframe(results_df, width="stretch", hide_index=True)

    ready = "results_df" in st.session_state
    if not ready:
        st.info("Run WCSS to continue.")
    if st.button("Continue", disabled=not ready, type="primary"):
        go_to(3)

# ---------------------------------------------------------------------------
# Step 3 — Choose k and create clusters
# ---------------------------------------------------------------------------
elif step == 3:
    st.header("Create the clusters")

    results_df = st.session_state["results_df"]
    k_bounds = (int(results_df["k"].min()), int(results_df["k"].max()))

    use_llm_for_k = st.checkbox("Use LLM to recommend k from WCSS instead of silhouette")

    if st.button("Auto-select k"):
        if use_llm_for_k:
            try:
                auto_k, reason = llm.recommend_k_from_wcss(results_df)
            except llm.LLMError as e:
                auto_k, reason = None, f"Auto-K failed: {e}"
        else:
            auto_k, reason = pipeline.auto_select_k_by_silhouette(results_df)

        if auto_k is not None:
            st.session_state["selected_k"] = auto_k
        st.info(reason)

    default_k = st.session_state.get("selected_k", k_bounds[0])
    selected_k = st.slider("Select k", min_value=k_bounds[0], max_value=k_bounds[1], value=default_k)

    if st.button("Create clusters"):
        data_scaled = st.session_state["data_scaled"]
        labels, fitted_kmeans = pipeline.run_kmeans(data_scaled, selected_k)
        summary = pipeline.build_cluster_summary(st.session_state["df_original"], labels)

        st.session_state["labels"] = labels
        st.session_state["fitted_kmeans"] = fitted_kmeans
        st.session_state["cluster_summary"] = summary

        os.makedirs("models", exist_ok=True)
        pipeline.save_model(fitted_kmeans, "models/kmeans_model.joblib")

    if "cluster_summary" in st.session_state:
        st.checkbox("Remove outliers before export", key="remove_outliers")
        if st.session_state.get("remove_outliers"):
            outlier_mask = pipeline.detect_outliers(
                st.session_state["data_scaled"],
                st.session_state["labels"],
                st.session_state["fitted_kmeans"],
            )
            st.session_state["outlier_mask"] = outlier_mask
            st.write(f"{int(outlier_mask.sum())} outlier row(s) will be removed on export.")
        else:
            st.session_state["outlier_mask"] = None

        st.caption("Cluster counts (names/descriptions are added in the next step)")
        display_cluster_table(st.session_state["cluster_summary"], show_name_description=False)

        st.subheader("Customer segments")
        feature_options = pipeline.numeric_feature_columns(st.session_state["df_original"])
        if len(feature_options) >= 2:
            by_variance = (
                st.session_state["df_original"][feature_options].var().sort_values(ascending=False).index.tolist()
            )
            fx, fy, fsize = st.columns(3)
            x_col = fx.selectbox("X axis", feature_options, index=feature_options.index(by_variance[0]))
            y_col = fy.selectbox(
                "Y axis", feature_options,
                index=feature_options.index(by_variance[1]) if len(by_variance) > 1 else 0,
            )
            size_col = fsize.selectbox(
                "Bubble size", feature_options,
                index=feature_options.index(by_variance[2]) if len(by_variance) > 2 else 0,
            )
            st.pyplot(
                pipeline.plot_customer_segments(
                    st.session_state["df_original"], st.session_state["labels"], x_col, y_col, size_col
                ),
                width="content",
            )
        else:
            st.info("Need at least 2 numeric columns to chart customer segments.")

    ready = "cluster_summary" in st.session_state
    if not ready:
        st.info("Create clusters to continue.")
    if st.button("Continue", disabled=not ready, type="primary"):
        go_to(4)

# ---------------------------------------------------------------------------
# Step 4 — LLM cluster naming
# ---------------------------------------------------------------------------
elif step == 4:
    st.header("Name the groups with an LLM")

    if st.button("Generate names/descriptions with LLaMA"):
        summary = st.session_state["cluster_summary"]
        skip_cols = {"cluster_id", "count", "name", "description"}
        numeric_cols = [
            c for c in summary.select_dtypes(include="number").columns if c not in skip_cols
        ]
        categorical_cols = [
            c for c in summary.select_dtypes(exclude="number").columns if c not in skip_cols
        ]
        try:
            model_name = llm.current_model_name()
            with st.spinner(f"Asking {model_name} to name {len(summary)} cluster(s)..."):
                updated_summary, errors = llm.generate_cluster_names(summary, numeric_cols, categorical_cols)
            st.session_state["cluster_summary"] = updated_summary
            for err in errors:
                st.warning(err)
        except llm.LLMError as e:
            st.error(str(e))

    st.caption("Cluster labels (with name + description)")
    display_cluster_table(st.session_state["cluster_summary"])

    if st.button("Continue", type="primary"):
        go_to(5)

# ---------------------------------------------------------------------------
# Step 5 — Export
# ---------------------------------------------------------------------------
elif step == 5:
    st.header("Export clustered CSV")

    summary = st.session_state["cluster_summary"]
    df_original = st.session_state["df_original"]
    labels = st.session_state["labels"]

    name_map = summary.set_index("cluster_id")["name"]
    description_map = summary.set_index("cluster_id")["description"]
    export_df = df_original.loc[labels.index].copy()
    export_df["name_cluster"] = labels.map(name_map).values
    export_df["description_cluster"] = labels.map(description_map).values

    outlier_mask = st.session_state.get("outlier_mask")
    if outlier_mask is not None:
        export_df = export_df.loc[~outlier_mask]

    filename = f"{st.session_state['uploaded_name']}_clustered.csv"
    st.download_button(
        "Download clustered CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )

    if st.button("Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        go_to(1)
