
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

df = pd.read_csv("Twitter_Data.csv", encoding='utf-8')
df = df[['clean_text','category']]

df.dropna(subset=['clean_text','category'], inplace=True)

df['category'] = df['category'].astype(int)

label_map = {-1:0, 0:1, 1:2}
df['label'] = df['category'].map(label_map)

print(df.head())
print(df['label'].value_counts())

def clean_text(text):
    text = re.sub(r'https?://\S+','', text)
    text = re.sub(r'@\w+','', text)
    text = re.sub(r'#','', text)
    return text.strip()

df['text_clean'] = df['clean_text'].astype(str).apply(clean_text)

X = df['text_clean'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative','Neutral','Positive']))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative','Neutral','Positive'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

sample_texts = [
    "I love this new product, it's amazing!",
    "This is okay, nothing special.",
    "I hate this experience, very bad service."
]
preds = pipeline.predict(sample_texts)
print("\nSample Predictions:")
for txt, pred in zip(sample_texts, preds):
    sentiment = ['Negative','Neutral','Positive'][pred]
    print(f"{txt} --> {sentiment}")

joblib.dump(pipeline, "sentiment_model.joblib")
print("Model saved as sentiment_model.joblib")
