import pandas as pd

data = pd.read_csv("dataset/IPL.csv")
print(sorted(data["venue"].dropna().unique()))
