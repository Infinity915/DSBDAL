import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
# Load the Iris dataset natively from scikit-learn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
df['species'] = iris.target_names[iris.target]
# Prepend 'Iris-' to match the typical dataset format
df['species'] = df['species'].apply(lambda x: 'Iris-' + x)

print("--- Iris Dataset Preview ---")
print(df.head())

# ==========================================
# STEP 2: Define Features (X) and Target (y)
# ==========================================
# Features are the sepal and petal measurements
X = df.drop('species', axis=1)
# Target is the species of the flower
y = df['species']

# ==========================================
# STEP 3: Split the Data
# ==========================================
# Split the dataset: 80% for Training, 20% for Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# STEP 4: Initialize and Train the Model
# ==========================================
# We use Gaussian Naive Bayes because our features are continuous numeric values
model = GaussianNB()
model.fit(X_train, y_train)

# ==========================================
# STEP 5: Make Predictions
# ==========================================
y_pred = model.predict(X_test)

# ==========================================
# STEP 6: Evaluate the Model (Confusion Matrix)
# ==========================================
# Generate the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate overall performance metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

# Because Iris has 3 classes (multi-class), we must use the 'macro' average 
# to calculate precision and recall across all classes.
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')

# Print the overall results
print("\n--- Confusion Matrix ---")
print(cm)
print("Labels order:", model.classes_)

print("\n--- Overall Model Evaluation Metrics ---")
print(f"Accuracy:   {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"Precision:  {precision:.2f}")
print(f"Recall:     {recall:.2f}")

# ==========================================
# STEP 7: Extract TP, FP, TN, FN for a specific class
# ==========================================
# The Iris dataset has 3 classes, so the confusion matrix is 3x3.
# We will isolate the metrics specifically for 'Iris-setosa' (which is at index 0).

TP_setosa = cm[0, 0]
FP_setosa = cm[:, 0].sum() - TP_setosa
FN_setosa = cm[0, :].sum() - TP_setosa
TN_setosa = cm.sum() - (TP_setosa + FP_setosa + FN_setosa)

print("\n--- Specific Breakdown for 'Iris-setosa' ---")
print(f"True Positives (TP): {TP_setosa}")
print(f"True Negatives (TN): {TN_setosa}")
print(f"False Positives (FP): {FP_setosa}")
print(f"False Negatives (FN): {FN_setosa}")