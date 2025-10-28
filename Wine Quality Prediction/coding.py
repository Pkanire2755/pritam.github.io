
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

df = pd.read_csv("WineQT.csv")

print("Shape:", df.shape)
print(df.info())
print(df.head())
print(df['quality'].value_counts())

print("\nMissing values:\n", df.isnull().sum())

df.drop_duplicates(inplace=True)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
print("✅ Best model saved as 'wine_quality_model.pkl'")
