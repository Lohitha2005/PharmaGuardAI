import pandas as pd

df = pd.read_csv("datasets/tox21.csv")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 3 rows:")
print(df.head(3))
print("\nNull counts:")
print(df.isnull().sum())
