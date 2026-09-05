import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL._imaging import display
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")


def get_normalization_csv():
    """
    this function gets the users CSV file by path,and normalize it at the same time
    saves a copy of the original csv
    """
    ##get the data
    while True:
        file_path = input('Please insert you csv file or path here\n').strip()
        print()
        file_path = file_path.strip('"\'')
        try:
            df = pd.read_csv(file_path)
            df_original = df.copy()
            # dropping every missing data & every column with id in it.
            df = df.drop(columns=df.filter(regex='(?i)id').columns)
            df = df.dropna()

            ##normaliztion of the data
            df = pd.get_dummies(df, drop_first=True, dtype=int)
            scaler = StandardScaler()
            data_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
            return df, df_original, data_scaled

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found. Please try again.\n")
        except Exception as e:
            print(f"An unexpected error occurred: {e}. Please try again.\n")


df, df_original, data_scaled = get_normalization_csv()
print("The normalized DataFrame:")
display(data_scaled)
print('==' * 35)
print()
print("The original DataFrame:")
display(df_original)


def get_k():
    """
    Prompts the user to enter valid maximum and minimum K values.
    Returns (user_choice_max, user_choice_min).
    """
    while True:
        try:
            user_choice_min = int(input('Please enter the lowest K: '))
            user_choice_max = int(input('Please enter the highest K: '))
            if user_choice_max <= user_choice_min:
                print('Error - the highest number is lower or equal to the lowest number')
                continue

            return user_choice_max, user_choice_min

        except ValueError:
            print('Error: Invalid input. Please enter numbers only')


user_choice_max, user_choice_min = get_k()

wcss = []

for i in range(user_choice_min, user_choice_max+1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

#saving it as new df
table_kmean_wcss = pd.DataFrame({
    'kmeans': range(user_choice_min, user_choice_max+1),
    'wcss': wcss})

display(table_kmean_wcss)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 5))

plt.plot(range(user_choice_min, user_choice_max+1), wcss, 'bx-', linewidth=2, markersize=8)

plt.title('The Elbow Method for Optimal K', fontsize=14, fontweight='bold')
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12)
plt.axvline(
    x=7, color="red", linestyle="--", linewidth=1.5, label="Optimal K = 7")
plt.legend()
plt.xticks(range(user_choice_min, user_choice_max+1))
plt.show()

def user_k():
    while True:
        try:
            user = int(input(f'Please choose your K from {user_choice_min} up to {user_choice_max}: '))
            if user > user_choice_max or user < user_choice_min:
                print('Error - the number you picked is not in range')
                continue
            return user
        except ValueError:
            print('Error: Invalid input. Please enter numbers only')

user = user_k()
kmeans = KMeans(n_clusters=user, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(data_scaled)
df_original['cluster'] = cluster_labels
print('value count:', df_original['cluster'].value_counts())

df_summary = df_original.groupby('cluster').mean(numeric_only=True).reset_index()
df_summary = df_summary.rename(columns={'cluster': 'cluster_id'})
df_summary['count'] = df_original['cluster'].value_counts().sort_index().values
df_summary['name'] = ""
df_summary['description'] = ""
df_summary