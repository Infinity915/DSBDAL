import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# STEP 1: Load the Dataset
# ==========================================
# Load the inbuilt 'titanic' dataset directly from Seaborn
titanic = sns.load_dataset('titanic')

print("--- Titanic Dataset Preview ---")
print(titanic.head())

# ==========================================
# STEP 2: Exploratory Data Analysis (Finding Patterns)
# ==========================================
# Let's find patterns by looking at survival rates based on Class and Gender
print("\n--- Survival Rate by Passenger Class ---")
print(titanic.groupby('pclass')['survived'].mean())

print("\n--- Survival Rate by Gender ---")
print(titanic.groupby('sex')['survived'].mean())

# Plotting a pattern: Survival by Gender and Class
plt.figure(figsize=(8, 5))
sns.barplot(x='pclass', y='survived', hue='sex', data=titanic)
plt.title('Survival Rate by Passenger Class and Gender')
plt.ylabel('Survival Probability')
plt.xlabel('Passenger Class (1=1st, 2=2nd, 3=3rd)')
plt.show()

# ==========================================
# STEP 3: Distribution of Ticket Prices (Fare)
# ==========================================
# We use histplot to see how the 'fare' prices are distributed
plt.figure(figsize=(10, 6))

# kde=True adds the smooth curve line over the bars
sns.histplot(titanic['fare'], kde=True, bins=40, color='blue')

plt.title("Distribution of Ticket Prices (Fare) on the Titanic")
plt.xlabel("Fare Price (£)")
plt.ylabel("Frequency (Number of Passengers)")
plt.xlim(0, 300) # Limiting X-axis to 300 to zoom in on the majority of data

plt.tight_layout()
plt.show()

# Print an observation about the histogram
print("\n--- Observation on Fare Distribution ---")
print("The histogram shows a heavily right-skewed distribution. The vast majority of passengers paid very low fares (under £50), representing the 3rd class passengers. Only a very small handful of wealthy individuals paid extreme outlier prices (over £100).")