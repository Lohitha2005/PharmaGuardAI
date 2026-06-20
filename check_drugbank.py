import pandas as pd
df = pd.read_csv("datasets/drugbank.tsv", sep="\t")
print(df.columns.tolist())
print(df.head(2))