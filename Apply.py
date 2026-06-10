import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix





df = pd.read_csv("heart.csv")
df.head()

df.isnull().sum()

df = pd.get_dummies(df, columns=['sex','cp','thal'], drop_first=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#EDA
sns.countplot(x='target', data=df)
plt.show()
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, annot_kws={'size': 8})
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#predictive modeling
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression / Random Forest
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))


