import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder


script_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(script_dir, "titanic_cleaned.csv"))

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare', 'FamilySize']] = scaler.fit_transform(df[['Age', 'Fare', 'FamilySize']])

df.to_csv(os.path.join(script_dir, "titanic_processed.csv"), index=False)

print("Data processing complete. File saved as titanic_processed.csv")
print(df.head())
