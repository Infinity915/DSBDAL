import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
# Load the dataset (replace 'purchase.csv' with your actual file path if needed)
df = pd.read_csv('purchase.csv')

print("--- Dataset Preview ---")
print(df.head())

# ==========================================
# STEP 2: Define Features (X) and Target (y)
# ==========================================
# We use 'Age' and 'EstimatedSalary' as our independent features
X = df[['Age', 'EstimatedSalary']]

# 'Purchased' is our target response variable (0 = No, 1 = Yes)
y = df['Purchased']

# ==========================================
# STEP 3: Split the Data
# ==========================================
# Split the dataset: 75% for Training, 25% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==========================================
# STEP 4: Feature Scaling (Crucial for Logistic Regression)
# ==========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================================
# STEP 5: Initialize and Train the Model
# ==========================================
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# ==========================================
# STEP 6: Make Predictions
# ==========================================
y_pred = model.predict(X_test)

# ==========================================
# STEP 7: Evaluate the Model (Confusion Matrix)
# ==========================================
# Generate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Extract TP, FP, TN, FN from the matrix using ravel()
TN, FP, FN, TP = cm.ravel()

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Print the results
print("\n--- Confusion Matrix ---")
print(cm)
print(f"\nTrue Positives (TP): {TP}")
print(f"True Negatives (TN): {TN}")
print(f"False Positives (FP): {FP}")
print(f"False Negatives (FN): {FN}")

print("\n--- Model Evaluation Metrics ---")
print(f"Accuracy:   {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision:  {precision:.2f}")
print(f"Recall:     {recall:.2f}")