import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
            return df, df_original, data_scaled

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Please try again.\n")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Please try again.\n")


# 1. Load and Normalize Data
df, df_original, data_scaled = get_normalization_csv()

# print("The normalized DataFrame:")
# print(data_scaled.head())
# print('==' * 35)
# print("\nThe original DataFrame:")
# print(df_original.head())


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


# 4. Final Cluster Selection & Summary Table
def user_k():
    while True:
        try:
            user = int(input(f'\nPlease choose your K from {user_choice_min} up to {user_choice_max}: '))
            if user > user_choice_max or user < user_choice_min:
                print('Error - the number you picked is not in range')
                continue
            return user
        except ValueError:
            print('Error: Invalid input. Please enter numbers only.')


user = user_k()
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

import requests

API_KEY = "90900d1782a6491aabab297e639c411d.R53bbUJPJmSxloC95Q7PHFlR"
URL = "https://ollama.com/api/chat"

def ask_llm(prompt):
    response = requests.post(URL,
        headers={
            "Authorization": f"Bearer {API_KEY}"
        },
        json={
            "model": "gpt-oss:120b",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
    )
    return response.json()["message"]["content"]

for id in df_summary['cluster_id']:
    df_summary.loc[id,'name'] = ask_llm(f'give me a short name between 1-3 words for this cluster:\n{df_summary.iloc[id]}')
    df_summary.loc[id,'description'] = ask_llm(f'give me a short sentence up to 10 words for this cluster:\n{df_summary.iloc[id]}')
print(df_summary)


import os

original_file_path = "customers.csv"

base_name = os.path.splitext(original_file_path)[0]

output_filename = f"{base_name}_clustered.csv"

df_summary.to_csv(output_filename, index=False)

print(f"הקובץ נשמר בהצלחה במחשב בשם: {output_filename}")
full_path = os.path.abspath(output_filename)
print(f"הקובץ נשמר בדיוק כאן:\n{full_path}")