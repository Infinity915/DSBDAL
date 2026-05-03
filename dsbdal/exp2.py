import pandas as pd
import numpy as np

# ==========================================
# 0. Create the "Academic Performance" Dataset
# ==========================================
# We are intentionally injecting missing values (np.nan), 
# text inconsistencies ('m', 'Male'), and extreme outliers (150, 10)
data = {
    'student_id': range(1, 16),
    'gender': ['M', 'F', 'Male', 'm', 'F', 'F', 'M', 'Male', 'F', 'M', 'm', 'F', 'M', 'F', 'M'],
    'math_score': [65, 78, np.nan, 63, 74, 41, 44, 61, 150, 75, 59, np.nan, 82, 68, 70],  # Note the 150 (Outlier)
    'reading_score': [68, 72, 69, 93, 75, np.nan, 83, 62, 77, 55, 10, 88, 79, np.nan, 81], # Note the 10 (Outlier)
    'placement_score': [94, 90, 78, 64, 94, 50, 74, 56, 58, 73, 85, 91, 88, 72, 80]
}
df = pd.DataFrame(data)

print("--- Initial Messy Dataset ---")
print(df.head(10))

# ==========================================
# 1. Scan for Missing Values & Inconsistencies
# ==========================================
print("\n--- 1A. Scanning for Missing Values ---")
print(df.isnull().sum())

# Handle Inconsistencies in Categorical Data ('Male', 'm' -> 'M')
df['gender'] = df['gender'].replace({'Male': 'M', 'm': 'M'})

# Handle Missing Numeric Values (Fill with Median/Mean)
df['math_score'] = df['math_score'].fillna(df['math_score'].median())
df['reading_score'] = df['reading_score'].fillna(df['reading_score'].mean())

print("\n--- 1B. Missing Values After Cleaning ---")
print(df.isnull().sum())


# ==========================================
# 2. Scan and Handle Outliers (IQR Method)
# ==========================================
# Function to cap outliers using the Interquartile Range (IQR)
def handle_outliers(dataframe, column):
    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Cap the extreme values to the upper and lower bounds
    dataframe[column] = np.where(dataframe[column] > upper_bound, upper_bound, dataframe[column])
    dataframe[column] = np.where(dataframe[column] < lower_bound, lower_bound, dataframe[column])
    return dataframe

# Apply the outlier handler to our score columns
df = handle_outliers(df, 'math_score')
df = handle_outliers(df, 'reading_score')

print("\n--- 2. Dataset After Capping Outliers ---")
print(df[['math_score', 'reading_score']].head(12))


# ==========================================
# 3. Data Transformations
# ==========================================
# We will apply a Logarithmic Transformation to 'math_score'
# Reason: To decrease right-skewness and bring the distribution closer to a normal curve.
# We use np.log1p (log(1 + x)) to safely handle any potential zeros.

df['math_score_log_transformed'] = np.log1p(df['math_score'])

print("\n--- 3. Data Transformation (Log Scale) ---")
print(df[['math_score', 'math_score_log_transformed']].head())