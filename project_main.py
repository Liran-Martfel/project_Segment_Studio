import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


def get_normalization_csv():
    """
    this function gets the users CSV file by path, and normalizes it at the same time.
    saves a copy of the original csv.
    """
    while True:
        file_path = input('Please insert your CSV file path here: ').strip()
        print()
        file_path = file_path.strip('"\'')
        try:
            df = pd.read_csv(file_path)
            df_original = df.copy()

            # dropping missing data & columns with 'id' in their name
            df = df.drop(columns=df.filter(regex='(?i)id').columns)
            df = df.dropna()

            # normalization of the data
            df = pd.get_dummies(df, drop_first=True, dtype=int)
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
            return df, df_original, data_scaled, file_path

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Please try again.\n")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Please try again.\n")


# 2. Get Range for K
def get_k():
    """
    Prompts the user to enter valid maximum and minimum K values.
    Returns (user_choice_max, user_choice_min).
    """
    while True:
        try:
            user_choice_min = int(input('\nPlease enter the lowest K: '))
            user_choice_max = int(input('Please enter the highest K: '))
            if user_choice_max <= user_choice_min:
                print('Error - the highest number must be greater than the lowest number.')
                continue

            return user_choice_max, user_choice_min

        except ValueError:
            print('Error: Invalid input. Please enter numbers only.')


# 4. Final Cluster Selection & Summary Table
def user_k(user_choice_min, user_choice_max):
    while True:
        try:
            user = int(input(f'\nPlease choose your K from {user_choice_min} up to {user_choice_max}: '))
            if user > user_choice_max or user < user_choice_min:
                print('Error - the number you picked is not in range')
                continue
            return user
        except ValueError:
            print('Error: Invalid input. Please enter numbers only.')


API_KEY = os.environ.get("OLLAMA_API_KEY", "")
URL = "https://ollama.com/api/chat"


def ask_llm(prompt, max_words=None):
    """
    Sends a prompt to the Ollama Cloud model and returns its text reply.
    If max_words is given, the reply is capped at max_words + 3 words (a
    small buffer for a slightly wordy answer) instead of the full response.
    Network/timeout/malformed-response errors are caught so one failed
    cluster doesn't crash the whole naming loop.
    """
    try:
        response = requests.post(URL,
            headers={
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": "gpt-oss:120b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        text = response.json()["message"]["content"].strip()
    except (requests.exceptions.RequestException, KeyError, ValueError) as e:
        print(f"LLM request failed: {e}")
        return "N/A"

    if max_words is not None:
        words = text.split()
        text = " ".join(words[: max_words + 3])

    return text


def main():
    # 1. Load and Normalize Data
    df, df_original, data_scaled, file_path = get_normalization_csv()

    # print("The normalized DataFrame:")
    # print(data_scaled.head())
    # print('==' * 35)
    # print("\nThe original DataFrame:")
    # print(df_original.head())

    user_choice_max, user_choice_min = get_k()

    # 3. Calculate WCSS and Plot Elbow Method
    wcss = []
    for i in range(user_choice_min, user_choice_max + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        wcss.append(kmeans.inertia_)

    table_kmean_wcss = pd.DataFrame({
        'kmeans': range(user_choice_min, user_choice_max + 1),
        'wcss': wcss
    })

    print("\nWCSS Table:")
    print(table_kmean_wcss)

    # Silhouette Scores Calculation
    print("\nCalculating Silhouette Scores:")
    for k in range(user_choice_min, user_choice_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data_scaled)
        score = silhouette_score(data_scaled, kmeans.labels_)
        print(f"Silhouette Score for K={k}: {score:.4f}")

    # sns.set_theme(style="whitegrid")
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(user_choice_min, user_choice_max + 1), wcss, 'bx-', linewidth=2, markersize=8)
    # plt.title('The Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    # plt.xlabel('Number of Clusters (K)', fontsize=12)
    # plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
    # plt.axvline(x=7, color="red", linestyle="--", linewidth=1.5, label="Optimal K = 7")
    # plt.xticks(range(user_choice_min, user_choice_max + 1))
    # plt.show()

    user = user_k(user_choice_min, user_choice_max)
    kmeans = KMeans(n_clusters=user, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_scaled)
    df_original['cluster'] = cluster_labels

    print('\nValue count per cluster:')
    print(df_original['cluster'].value_counts())

    # Build Summary Table
    df_summary = df_original.groupby('cluster').mean(numeric_only=True).reset_index()
    df_summary = df_summary.rename(columns={'cluster': 'cluster_id'})
    df_summary['count'] = df_original['cluster'].value_counts().sort_index().values
    df_summary['name'] = ""
    df_summary['description'] = ""

    print("\nSummary DataFrame:")
    print(df_summary)

    for _, row in df_summary.iterrows():
        cluster_id = row['cluster_id']
        df_summary.loc[df_summary['cluster_id'] == cluster_id, 'name'] = ask_llm(
            f'give me a short name between 1-3 words for this cluster:\n{row}', max_words=3
        )
        df_summary.loc[df_summary['cluster_id'] == cluster_id, 'description'] = ask_llm(
            f'give me a short sentence up to 10 words for this cluster:\n{row}', max_words=10
        )
    print(df_summary)

    original_file_path = file_path
    base_name = os.path.splitext(original_file_path)[0]
    output_filename = f"{base_name}_clustered.csv"

    df_summary.to_csv(output_filename, index=False)

    print(f"הקובץ נשמר בהצלחה במחשב בשם: {output_filename}")
    full_path = os.path.abspath(output_filename)
    print(f"הקובץ נשמר בדיוק כאן:\n{full_path}")


if __name__ == "__main__":
    main()
