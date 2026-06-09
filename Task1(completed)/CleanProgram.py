import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(script_dir, "Titanic.csv"))  

print("Initial Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(columns=['Cabin'])
df = df.drop_duplicates()

df['Survived'] = df['Survived'].astype(int)
df.to_csv(os.path.join(script_dir, "titanic_cleaned.csv"), index=False)

print("Cleaned Shape:", df.shape)
print("Data cleaning complete. File saved as titanic_cleaned.csv")