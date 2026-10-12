from pathlib import Path
import pandas as pd

DATA_Path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(DATA_Path)

print("\n-- Dataset Shape -- ")
print(df.shape)

print("\n-- First Five rows --")
print(df.head())

print("\n-- Column Name and data types --")
print(df.dtypes)

print("\n-- Missing Values --")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\n-- Churn distribution --")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3))

print("\n-- TotalCharges inspection --")
print("Data type:", df['TotalCharges'].dtypes)
print("Blank Strings:", df['TotalCharges'].astype(str).str.strip().eq("").sum())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")

print("Missing TotalCharges: ",df['TotalCharges'].isna().sum())

print("\n-- All Column with missing values --")
print(df.isna().sum()[df.isna().sum() > 0])

print("\n-- All data types --")
print(df.dtypes.to_string())