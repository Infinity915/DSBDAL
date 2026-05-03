import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('dataset.csv')

df['Age']=df['Age'].fillna(df['Age'].mean())
df['Department']=df['Department'].fillna(df['Department'].mode()[0])
df['Salary']=df['Salary'].fillna(df['Salary'].median())

df['ID']=df['ID'].astype(int)
df['Age']=df['Age'].astype(int)
df['Salary']=df['Salary'].astype(int)
df['Department']=df['Department'].astype(str)

print("This is the preprocessed data (missing values handled, whole numbers):")
print(df)

print("\n" + "="*100)
print("This is the normalized data (Min-Max Scaling 0-1):")
print("="*100)

df_normalized=df.copy()
scaler=MinMaxScaler()
df_normalized['Age','Salary']=scaler.fit_transform(df[['Age','Salary']])
print(df_normalized)

