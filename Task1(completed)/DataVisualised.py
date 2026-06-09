import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(script_dir, "titanic_processed.csv"))

sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

sns.histplot(df["Fare"], bins=20, kde=True)
plt.title("Fare Distribution")
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={'label': 'Correlation'})
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
