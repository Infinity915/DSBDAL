import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
# Read the Boston Housing dataset. 
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
boston_df = pd.read_csv(url)

print("--- Boston Housing Dataset Preview ---")
print(boston_df.head())

# ==========================================
# STEP 2: Define Features (X) and Target (y)
# ==========================================
# 'medv' is the target variable we want to predict (Median value of homes in $1000s)
# The remaining 13 columns are our independent features (predictors)
X = boston_df.drop('medv', axis=1)
y = boston_df['medv']

# ==========================================
# STEP 3: Split the Data
# ==========================================
# Split the dataset into a Training Set (80%) and a Testing Set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Split Info ---")
print(f"Total samples: {len(boston_df)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ==========================================
# STEP 4: Initialize and Train the Model
# ==========================================
# Create the Linear Regression object
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# ==========================================
# STEP 5: Make Predictions
# ==========================================
# Predict the house prices for the testing set
y_pred = model.predict(X_test)

# ==========================================
# STEP 6: Evaluate Model Performance
# ==========================================
# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation Metrics ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-Squared Score (Accuracy): {r2:.2f}")