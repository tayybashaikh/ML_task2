# Task 2: Data Cleaning & Missing Value Handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load Dataset
df = pd.read_csv("KaggleV2-may-2016.csv")
print("FIRST 5 ROWS:\n", df.head())
print("\nDataset Info:")
print(df.info())

# 2️⃣ Check Missing Values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# 3️⃣ Visualize Missing Values (Optional)
plt.figure(figsize=(8,4))
df.isnull().sum().plot(kind='bar', title='Missing Values per Column')
plt.show()

# 4️⃣ Handle Missing Values
# Numerical columns → median imputation
num_cols = df.select_dtypes(include=['float64','int64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical columns → mode imputation
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 5️⃣ Drop columns with extremely high missing values (threshold 50%)
threshold = 0.5
df = df[df.columns[df.isnull().mean() < threshold]]

# 6️⃣ Validate cleaned dataset
print("\nAfter Cleaning - Missing Values per Column:")
print(df.isnull().sum())
print("\nDataset Shape after Cleaning:", df.shape)

# 7️⃣ Save Cleaned Dataset
df.to_csv("Medical_Appointment_Cleaned.csv", index=False)
print("\nCleaned dataset saved as 'Medical_Appointment_Cleaned.csv'")