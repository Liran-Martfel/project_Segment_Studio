# 🧩 Segment Studio — פילוח אוטומטי (Clustering)

### 🌐 [נסו את האפליקציה החיה](https://projectsegmentstudio-a7yvlmvqjcnkntaeylnkgv.streamlit.app/)
*(דורש התחברות חד-פעמית וקצרה עם Google/GitHub — מדיניות של Streamlit Community Cloud לכל צופה, גם באפליקציות ציבוריות. שלב 4 (שם/תיאור בעזרת LLM) פועל רק בהרצה מקומית, כי הוא תלוי ב-Ollama שרץ על המחשב המריץ את האפליקציה.)*

### 🔗 [רשימת המשימות האינטראקטיבית של הפרויקט](https://claude.ai/code/artifact/600c6f71-059c-4354-8c04-15dfa6b86600)

מערכת אינטראקטיבית לניתוח נתונים שמטרתה לגלות קבוצות נסתרות (Segments) בתוך קובץ CSV באמצעות למידת מכונה.
המשתמש מעלה קובץ נתונים, בוחן כיצד הנתונים מתחלקים לקבוצות בעזרת אלגוריתם **K-Means**, ולאחר מכן משתמש במודל **LLM** כדי להעניק לכל קבוצה שם ותיאור קצר. המערכת נבנתה כאפליקציית **Streamlit** לעבודה אינטראקטיבית ונוחה עם הנתונים.

---

## 🔁 תהליך העבודה מקצה לקצה

האפליקציה בנויה כ-Wizard בן 5 שלבים עם פס התקדמות לחיצ ה (ניתן לדלג בין שלבים שכבר בוצעו):

1. **טעינת נתונים** — העלאת קובץ CSV והצגתו כטבלה.
2. **ניתוח מבנה הקבוצות (Elbow / WCSS)** — הרצת K-Means על טווח ערכי K, חישוב Silhouette Score לצדו, גרף Elbow עם קו מקווקו המסמן את ה-K המומלץ, וטבלת התוצאות המלאה.
3. **חישוב קלאסטרים** — הרצת K-Means עם ה-K הנבחר (או Auto-K), הקצאת `cluster_id` לכל שורה, ספירת תצפיות, אפשרות להסרת חריגים (Outliers), וגרף "Customer Segments" (בועות) על 3 פיצ'רים אמיתיים שהמשתמש בוחר.
4. **פרשנות בעזרת LLM** — לכל קלאסטר נבנה סיכום (גודל הקבוצה, ממוצעי פיצ'רים מספריים, ערכים קטגוריאליים שכיחים), שנשלח ל-Ollama המקומי ומחזיר שם קצר ותיאור בן שורה אחת.
5. **ייצוא לקובץ CSV** — הוספת עמודות `name_cluster` ו-`description_cluster` לקובץ המקורי ושמירתו בשם `<original_name>_clustered.csv`.

---

## 📁 מבנה הריפוזיטורי

> **מה זו ההגשה בפועל:** `app.py` (+ `src/pipeline.py`, `src/llm.py`) הן אפליקציית ה-Streamlit הסופית ומהוות את ההגשה.
> `project_main.py`, `Project segment studio.ipynb` ו-`checklist.ipynb`/`checklist.html` הם שלבי הפיתוח וההתקדמות שקדמו לאפליקציה — נשמרו בכוונה כתיעוד היסטוריה של הפרויקט, אך אינם חלק מהאפליקציה עצמה.

```
app.py                           - אפליקציית Streamlit הראשית (ה-Wizard בן 5 השלבים)
src/pipeline.py                  - לוגיקת ה-ML: נירמול, WCSS/Silhouette, K-Means, סיכום קלאסטרים, חריגים, גרפים
src/llm.py                       - קריאה ל-Ollama מקומי לשם/תיאור לכל קלאסטר + בחירת K אוטומטית
models/kmeans_model.joblib        - המודל המאומן האחרון שנשמר (נוצר בהרצה, מוגש כחלק מהפרויקט)
requirements.txt, .streamlit/    - תלויות והגדרות עיצוב (Theme סגול, ערכת נושא בהירה)
Mall_Customers.csv, customers_clustered.csv - קבצי דוגמה להרצה מהירה של האפליקציה
project02_clusters_dec25.pdf     - מסמך ההנחיות הרשמי של הפרויקט
Project segment studio.ipynb     - הנוטבוק המקורי (טעינה/נירמול/Elbow) ששימש לפיתוח הלוגיקה
project_main.py                  - סקריפט מקורי מבוסס input(), שממנו חולצה הלוגיקה ל-src/pipeline.py
checklist.ipynb / checklist.html - רשימת המשימות האינטראקטיבית של הפרויקט
```

## ⚙️ הרצה מקומית

1. התקנת תלויות:
   ```
   pip install -r requirements.txt
   ```
2. התקנת [Ollama](https://ollama.com) והורדת מודל מקומי (לשלב 4):
   ```
   ollama pull llama3
   ```
3. הרצת האפליקציה:
   ```
   streamlit run app.py
   ```
