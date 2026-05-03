import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
# Load the inbuilt 'titanic' dataset directly from Seaborn
titanic = sns.load_dataset('titanic')

# ==========================================
# STEP 2: Create the Box Plot
# ==========================================
plt.figure(figsize=(10, 6))

# Creating the boxplot
# x = Categorical grouping (Gender)
# y = Continuous numerical data (Age)
# hue = Secondary grouping (Survived: 0 = No, 1 = Yes)
sns.boxplot(x='sex', y='age', hue='survived', data=titanic, palette='Set2')

# Adding titles and labels
plt.title("Distribution of Age by Gender and Survival Status")
plt.xlabel("Gender (Sex)")
plt.ylabel("Age (Years)")

# Update the legend to be more readable
plt.legend(title='Survived', labels=['No (0)', 'Yes (1)'])

# Show the plot
plt.tight_layout()
plt.show()