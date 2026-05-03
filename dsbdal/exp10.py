import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # <-- FIXED: It must be pyplot
from sklearn.datasets import load_iris

# ==========================================
# STEP 0: Load the Dataset
# ==========================================
# Load the Iris dataset natively from scikit-learn
iris_data = load_iris()
iris = pd.DataFrame(data=iris_data.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
iris['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)

# ==========================================
# STEP 1: Features and Data Types
# ==========================================
print("--- 1. Features and Data Types ---")
print(iris.dtypes)

# ==========================================
# STEP 2: Histograms for Feature Distributions
# ==========================================
# Create a 2x2 grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Histograms of Iris Features', fontsize=16)

# Plot each feature with a Kernel Density Estimate (KDE) line
sns.histplot(iris['sepal_length'], kde=True, color='skyblue', ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length')

sns.histplot(iris['sepal_width'], kde=True, color='olive', ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width')

sns.histplot(iris['petal_length'], kde=True, color='gold', ax=axes[1, 0])
axes[1, 0].set_title('Petal Length')

sns.histplot(iris['petal_width'], kde=True, color='teal', ax=axes[1, 1])
axes[1, 1].set_title('Petal Width')

plt.tight_layout()
plt.show()

# ==========================================
# STEP 3: Box Plots for Distributions and Outliers
# ==========================================
plt.figure(figsize=(12, 8))
# We drop the 'species' column because boxplots require numeric data.
# Seaborn automatically plots a box for every remaining column.
sns.boxplot(data=iris.drop('species', axis=1), palette='Set2')

plt.title('Box Plots of Iris Numeric Features (Distribution & Outliers)', fontsize=16)
plt.ylabel('Measurement (cm)')
plt.xticks(rotation=45) # Rotate x-axis labels for better readability

plt.tight_layout()
plt.show()