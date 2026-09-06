"""Local Ollama integration for cluster naming and the WCSS-based auto-K bonus.

Uses a local Ollama install (http://localhost:11434) instead of a cloud API
key, so there is no secret to manage or leak.
"""

import json

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3"


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


def ask_llm_json(prompt: str, timeout: int = 30) -> dict:
    """Ask the local Ollama model a question, expecting a JSON object back."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                # Cap output length and lower temperature: the answer is one
                # small JSON object, so this cuts generation time and makes
                # the model less likely to add unparseable rambling text.
                "options": {"num_predict": 120, "temperature": 0.3},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
    except requests.exceptions.ConnectionError as e:
        raise LLMError(
            "Could not connect to Ollama at http://localhost:11434 — is it running?"
        ) from e
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
