"""Ollama integration for cluster naming and the WCSS-based auto-K bonus.

Uses local Ollama (http://localhost:11434) when available -- free, no key
needed, works when running the app on your own machine. Falls back to
Ollama Cloud when an OLLAMA_API_KEY is set (e.g. via Streamlit secrets on
a cloud deployment, where there's no local Ollama to reach), so the LLM
features work in both places without ever committing a key to git.
"""

import json
import os

import pandas as pd
import requests

OLLAMA_LOCAL_URL = "http://localhost:11434/api/chat"
OLLAMA_LOCAL_MODEL = "llama3"

OLLAMA_CLOUD_URL = "https://ollama.com/api/chat"
OLLAMA_CLOUD_MODEL = "gpt-oss:120b"


def _get_backend() -> tuple[str, str, str | None]:
    """Returns (url, model, api_key). api_key is None when using local Ollama."""
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        return OLLAMA_CLOUD_URL, OLLAMA_CLOUD_MODEL, api_key
    return OLLAMA_LOCAL_URL, OLLAMA_LOCAL_MODEL, None


def current_model_name() -> str:
    """Which model is actually in use right now (for display, e.g. a spinner message)."""
    return _get_backend()[1]


class LLMError(Exception):
    """Raised when Ollama can't be reached or returns something unusable."""


def _extract_json(text: str) -> dict:
    """Find the first balanced {...} block in text and parse it as JSON.

    More robust than a greedy regex: a model that rambles extra braces
    around the answer (markdown fences, asides) won't make a naive
    `\\{.*\\}` grab the wrong span.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)

    raise LLMError(f"Could not parse JSON from LLM response: {text!r}")


def ask_llm_json(prompt: str, timeout: int = 60) -> dict:
    """Ask Ollama (local, or Cloud if OLLAMA_API_KEY is set) for a JSON object back."""
    url, model, api_key = _get_backend()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    # gpt-oss:120b (the Cloud model) spends tokens on internal reasoning
    # before its visible answer, so it needs a much bigger budget than
    # llama3 (local) or its JSON gets cut off mid-object.
    num_predict = 500 if api_key else 120

    try:
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Cap output length and lower temperature: the answer is one
                # small JSON object, so this cuts generation time and makes
                # the model less likely to add unparseable rambling text.
                "options": {"num_predict": num_predict, "temperature": 0.3},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
    except requests.exceptions.ConnectionError as e:
        raise LLMError(f"Could not connect to Ollama at {url} — is it running?") from e
    except requests.exceptions.Timeout as e:
        raise LLMError("Ollama request timed out.") from e
    except (KeyError, ValueError) as e:
        raise LLMError(f"Unexpected response from Ollama: {e}") from e

    return _extract_json(content)


def _cluster_prompt(row: pd.Series, numeric_cols: list, categorical_cols: list) -> str:
    lines = [f"Cluster with {int(row['count'])} observations.", "Numeric feature means:"]
    for col in numeric_cols:
        lines.append(f"  {col}: {row[col]:.2f}")
    if categorical_cols:
        lines.append("Most frequent categorical values:")
        for col in categorical_cols:
            lines.append(f"  {col}: {row[col]}")
    lines.append(
        "\nRespond with ONLY a JSON object of the form "
        '{"name": "short cluster name (1-3 words)", '
        '"description": "one sentence up to 10 words"}. No extra text.'
    )
    return "\n".join(lines)


def generate_cluster_names(
    df_summary: pd.DataFrame, numeric_cols: list, categorical_cols: list
) -> tuple[pd.DataFrame, list]:
    """Fill in name/description for each cluster via the LLM.

    A failure on one cluster doesn't abort the rest — errors are collected
    and returned so the caller can surface them without losing progress on
    the clusters that succeeded.
    """
    df_summary = df_summary.copy()
    errors = []

    for idx, row in df_summary.iterrows():
        prompt = _cluster_prompt(row, numeric_cols, categorical_cols)
        result = None
        last_error: LLMError | None = None

        for _ in range(2):  # one retry — a single malformed response shouldn't blank out a cluster
            try:
                result = ask_llm_json(prompt)
                break
            except LLMError as e:
                last_error = e

        if result is not None:
            df_summary.loc[idx, "name"] = result.get("name", "")
            df_summary.loc[idx, "description"] = result.get("description", "")
        else:
            errors.append(f"Cluster {row['cluster_id']}: {last_error}")

    return df_summary, errors


def recommend_k_from_wcss(results_df: pd.DataFrame) -> tuple[int, str]:
    """Bonus: ask the LLM for the elbow point when silhouette isn't available."""
    table_text = results_df[["k", "wcss"]].to_string(index=False)
    prompt = (
        "Given this WCSS (inertia) table for different values of k in K-Means "
        "clustering, identify the 'elbow' point — the k after which WCSS "
        "stops decreasing significantly.\n\n"
        f"{table_text}\n\n"
        'Respond with ONLY a JSON object: {"k": <integer>, "reason": "short explanation"}.'
    )
    result = ask_llm_json(prompt)
    k = int(result["k"])
    reason = result.get("reason", "")
    return k, f"Auto-selected k={k} based on WCSS elbow (LLM): {reason}"
