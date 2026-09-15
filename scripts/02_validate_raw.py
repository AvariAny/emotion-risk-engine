import pandas as pd

df = pd.read_csv("data/raw/label0/books/claude.csv")

print(df.shape)
print(df.tail(3))
print(df.iloc[-1])