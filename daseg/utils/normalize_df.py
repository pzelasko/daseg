import pandas as pd
import os, sys

df_path = sys.argv[1]


df = pd.read_csv(df_path, sep=',')
df = df.set_index('Punct')

df_norm = df.div(df.sum(axis=1), axis=0)

print(df)
print(df_norm)





