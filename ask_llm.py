import requests
from project_main import df_summary

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

