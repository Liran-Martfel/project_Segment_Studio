"""K-Means clustering pipeline used by app.py.

Mirrors the logic in project_main.py, but as pure functions that take/return
values instead of reading from input() or calling plt.show(), so they can be
called from Streamlit.
"""

import joblib
import matplotlib.figure
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Above this many rows, silhouette_score (O(n^2)) is computed on a random
# sample instead of the full dataset -- negligible accuracy loss, huge speedup.
SILHOUETTE_SAMPLE_THRESHOLD = 2000
SILHOUETTE_SAMPLE_SIZE = 1000

ACCENT_COLOR = "#7C3AED"

# Fixed-order categorical palette (validated CVD-safe adjacent ordering).
# Scatter/bubble charts are an all-pairs form, so only the first 3 slots are
# validated CVD-safe against every other slot simultaneously; beyond that,
# adjacent pairs are still safe but a legend is required regardless (added
# below) since identity is never carried by color alone.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]


def load_and_normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean and normalize an already-loaded CSV DataFrame.

    Returns (df_encoded, df_original, data_scaled):
      - df_encoded: id-columns dropped, NA rows dropped, one-hot encoded
      - df_original: a copy of the input, unmodified
      - data_scaled: df_encoded standardized (mean 0, std 1)
    """
    df_original = df.copy()

    df_encoded = df.drop(columns=df.filter(regex="(?i)id").columns)
    df_encoded = df_encoded.dropna()

    df_encoded = pd.get_dummies(df_encoded, drop_first=True, dtype=int)
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(
        scaler.fit_transform(df_encoded), columns=df_encoded.columns, index=df_encoded.index
    )

    return df_encoded, df_original, data_scaled


def compute_wcss_silhouette(data_scaled: pd.DataFrame, k_min: int, k_max: int) -> pd.DataFrame:
    """Run K-Means for each k in [k_min, k_max] and collect WCSS + silhouette score.

    n_init="auto" lets scikit-learn pick a single k-means++ run (its seeding
    already avoids bad local minima), instead of always repeating the fit 10
    times -- ~50x faster here with a negligible inertia difference, and this
    sweep is only used to compare k values, not as the final answer.
    """
    n_samples = len(data_scaled)
    sample_size = (
        SILHOUETTE_SAMPLE_SIZE if n_samples > SILHOUETTE_SAMPLE_THRESHOLD else None
    )

    rows = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(data_scaled)
        score = silhouette_score(
            data_scaled, kmeans.labels_, sample_size=sample_size, random_state=42
        )
        rows.append({"k": k, "wcss": kmeans.inertia_, "silhouette": score})
    return pd.DataFrame(rows)


def _abbreviate_number(value: float, _pos=None) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def plot_elbow(results_df: pd.DataFrame, best_k: int | None = None) -> matplotlib.figure.Figure:
    """Build the Elbow plot (k on X, WCSS on Y) as a Figure, for st.pyplot().

    Y-axis ticks are abbreviated (12000 -> 12K) since WCSS grows with both
    row count and feature count and can otherwise print unreadably long
    numbers. An optional dashed line marks the recommended k.
    """
    fig = matplotlib.figure.Figure(figsize=(6, 3.3))
    ax = fig.subplots()
    ax.plot(
        results_df["k"], results_df["wcss"],
        marker="o", markersize=5, linewidth=1.5, color=ACCENT_COLOR, label="WCSS",
    )
    if best_k is not None:
        ax.axvline(x=best_k, color="black", linestyle="--", linewidth=1.2, label=f"Best k = {best_k}")

    ax.set_title("The Elbow Method for Optimal K", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=9)
    ax.set_ylabel("WCSS", fontsize=9)
    ax.set_xticks(results_df["k"])
    ax.yaxis.set_major_formatter(FuncFormatter(_abbreviate_number))
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def numeric_feature_columns(df_original: pd.DataFrame) -> list[str]:
    """Numeric columns usable as chart axes (id-like columns excluded)."""
    df = df_original.drop(columns=df_original.filter(regex="(?i)id").columns, errors="ignore")
    return list(df.select_dtypes(include="number").columns)


def plot_customer_segments(
    df_original: pd.DataFrame, labels: pd.Series, x_col: str, y_col: str, size_col: str
) -> matplotlib.figure.Figure:
    """Bonus visual: customer segments plotted on 2 real, user-chosen features.

    A 3rd chosen feature controls bubble size, so the chart stays a plain
    readable 2D scatter (unlike a PCA projection, which can blur separation
    when the data has many one-hot categorical columns) while still working
    for any CSV, since the columns are picked at call time, not hardcoded.
    """
    df = df_original.loc[labels.index]
    sizes = df[size_col].astype(float)
    size_span = sizes.max() - sizes.min()
    marker_sizes = 20 + (sizes - sizes.min()) / size_span * 260 if size_span > 0 else pd.Series(80, index=sizes.index)

    fig = matplotlib.figure.Figure(figsize=(6, 3.3))  # match plot_elbow's size
    ax = fig.subplots()

    for i, cluster_id in enumerate(sorted(labels.unique())):
        color = CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)]
        mask = labels.values == cluster_id
        ax.scatter(
            df.loc[mask, x_col], df.loc[mask, y_col],
            s=marker_sizes[mask], color=color, alpha=0.6, edgecolors="white", linewidths=0.5,
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel(x_col, fontsize=9)
    ax.set_ylabel(y_col, fontsize=9)
    ax.set_title(f"Customer Segments (bubble size = {size_col})", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def run_kmeans(data_scaled: pd.DataFrame, k: int) -> tuple[pd.Series, KMeans]:
    """Fit K-Means with the chosen k. Returns (cluster labels, fitted model)."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(data_scaled)
    return pd.Series(labels, index=data_scaled.index, name="cluster"), kmeans


def _most_frequent(series: pd.Series):
    mode = series.mode()
    return mode.iloc[0] if not mode.empty else None


def build_cluster_summary(df_original: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Build the cluster_id/count/name/description summary table.

    Numeric columns are summarized by mean, categorical (object) columns by
    their most frequent value per cluster.
    """
    df = df_original.loc[labels.index].copy()
    df["cluster"] = labels.values

    numeric_cols = df.select_dtypes(include="number").columns.drop("cluster", errors="ignore")
    categorical_cols = df.select_dtypes(exclude="number").columns

    summary = df.groupby("cluster")[list(numeric_cols)].mean()

    if len(categorical_cols):
        cat_summary = df.groupby("cluster")[list(categorical_cols)].agg(_most_frequent)
        summary = summary.join(cat_summary)

    summary = summary.reset_index().rename(columns={"cluster": "cluster_id"})
    summary["count"] = labels.value_counts().sort_index().values
    summary["name"] = ""
    summary["description"] = ""

    cols = ["cluster_id", "count"] + [c for c in summary.columns if c not in ("cluster_id", "count", "name", "description")] + ["name", "description"]
    return summary[cols]


def save_model(fitted_kmeans: KMeans, path: str) -> None:
    """Persist the fitted K-Means model to disk (joblib) for submission."""
    joblib.dump(fitted_kmeans, path)


def auto_select_k_by_silhouette(results_df: pd.DataFrame) -> tuple[int, str]:
    """Bonus: pick the k with the highest silhouette score."""
    best_row = results_df.loc[results_df["silhouette"].idxmax()]
    k = int(best_row["k"])
    reason = f"Auto-selected k={k} based on highest silhouette score: {best_row['silhouette']:.2f}"
    return k, reason


def detect_outliers(
    data_scaled: pd.DataFrame, labels: pd.Series, fitted_kmeans: KMeans, n_std: float = 2.0
) -> pd.Series:
    """Bonus: flag points whose distance to their cluster centroid is an outlier.

    A point is flagged if its distance from the centroid it was assigned to
    is more than n_std standard deviations above the mean distance within
    that same cluster.
    """
    centroids = fitted_kmeans.cluster_centers_
    assigned_centroids = centroids[labels.values]
    distances = ((data_scaled.values - assigned_centroids) ** 2).sum(axis=1) ** 0.5
    distances = pd.Series(distances, index=data_scaled.index)

    thresholds = distances.groupby(labels.values).transform(
        lambda d: d.mean() + n_std * d.std()
    )
    return distances > thresholds
