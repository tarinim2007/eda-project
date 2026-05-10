import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

df = df.drop_duplicates()

df = df.fillna(df.select_dtypes(include=np.number).mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

if len(numeric_cols) >= 2:
    for i in range(len(numeric_cols)-1):
        plt.figure()
        sns.scatterplot(x=df[numeric_cols[i]], y=df[numeric_cols[i+1]])
        plt.xlabel(numeric_cols[i])
        plt.ylabel(numeric_cols[i+1])
        plt.show()

corr = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

print("\nCorrelation Matrix:")
print(corr)

sns.pairplot(df)
plt.show()