import pandas as pd

# ==========================================
# 0. Load the Dataset
# ==========================================
# We load the Iris dataset directly from the UCI Machine Learning web repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)

# ==========================================
# PART 1: Summary Statistics Grouped by a Categorical Variable
# ==========================================
print("--- PART 1: Summary Statistics Grouped by Category ---")

# Grouping by 'species' (categorical) and calculating stats for 'sepal_length' (numeric)
grouped_stats = df.groupby('species')['sepal_length'].agg(['mean', 'median', 'min', 'max', 'std'])
print("\nSummary Statistics of Sepal Length grouped by Species:")
print(grouped_stats)

# Create a list/dictionary containing a numeric value for each response to the categorical variable
species_list = df['species'].unique().tolist()
species_mapping = {species: i for i, species in enumerate(species_list)}

# Create a new column applying this mapping
df['species_numeric'] = df['species'].map(species_mapping)

print("\nNumeric Value Mapping for Categorical Variable (Species):")
print(species_mapping)


# ==========================================
# PART 2: Basic Statistical Details of Species
# ==========================================
print("\n--- PART 2: Basic Statistical Details of Each Species ---")

# 1. Filter and describe Iris-setosa
print("\nStatistical details for 'Iris-setosa':")
setosa = df[df['species'] == 'Iris-setosa']
print(setosa.describe())

# 2. Filter and describe Iris-versicolor
print("\nStatistical details for 'Iris-versicolor':")
versicolor = df[df['species'] == 'Iris-versicolor']
print(versicolor.describe())

# 3. Filter and describe Iris-virginica
print("\nStatistical details for 'Iris-virginica':")
virginica = df[df['species'] == 'Iris-virginica']
print(virginica.describe())