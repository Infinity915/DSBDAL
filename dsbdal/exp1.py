import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
df = pd.read_csv('Placement.csv')

print("--- Initial Data Preview ---")
print(df.head())

# Columns to normalize (numeric features)
cols_to_scale = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']

# ==========================================
# STEP 2: Data Preprocessing
# ==========================================
print("\n--- 1. Dimensions of the Dataset (Rows, Columns) ---")
print(df.shape)

print("\n--- 2. Variable Descriptions ---")
print(df.info())

print("\n--- 3. Missing Values Count ---")
print(df.isnull().sum())

print("\n--- 4. Initial Statistics ---")
print(df.describe())

# ==========================================
# STEP 3: Data Formatting & Cleaning
# ==========================================
# Fill categorical columns with mode
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])
df['workex'] = df['workex'].fillna(df['workex'].mode()[0])
df['status'] = df['status'].fillna(df['status'].mode()[0])

# Fill numeric columns with mean
for col in ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']:
    df[col] = df[col].fillna(df[col].mean())

# Fill salary with 0 if missing, then cast to int
df['salary'] = df['salary'].fillna(0).astype(int)

print("\n--- 5. Data Types After Formatting ---")
print(df.dtypes)

# ==========================================
# STEP 4: Data Normalization
# ==========================================
scaler = MinMaxScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

print("\n--- 6. Data After Min-Max Normalization ---")
print(df[cols_to_scale].head())

# ==========================================
# STEP 5: Turn Categorical into Quantitative
# ==========================================
encoder = LabelEncoder()
df['gender'] = encoder.fit_transform(df['gender'])
df['workex'] = encoder.fit_transform(df['workex'])
df['status'] = encoder.fit_transform(df['status'])

# Handle missing values in nominal categories before encoding
df['hsc_s'] = df['hsc_s'].fillna(df['hsc_s'].mode()[0])
df['degree_t'] = df['degree_t'].fillna(df['degree_t'].mode()[0])

# One-hot encoding for nominal categories
df = pd.get_dummies(df, columns=['hsc_s', 'degree_t'], drop_first=True)

print("\n--- 7. Final Cleaned & Encoded Dataset ---")
print(df.head())
