import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(script_dir, "Titanic.csv"))
print(df.head())
print(df.info())
print(df.describe())

#univariate analysis

sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

#bivariate analysis
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.show()

#key factors affecting survival
# Select only numeric columns for correlation
df_numeric = df.select_dtypes(include=['number'])
corr = df_numeric.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
