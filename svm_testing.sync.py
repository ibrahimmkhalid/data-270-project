# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
# import json
# import gzip
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
# import re

# %%
data_path = "./data/combined.csv"
df = pd.read_csv(data_path, low_memory=False)
print(df.isna().sum() / len(df) * 100)

# %%
print(df.isna().sum())

# %%
shpae = df.shape
print(df.isna().sum().sum() / (shpae[0] * shpae[1]) * 100)

# %%
df.head()

# %%
df = df[["overall", "reviewText", "summary", "verified"]]

# %%
print(df.isna().sum() / len(df) * 100)

# %%
df = df.dropna()

# %%
print(df.isna().sum() / len(df) * 100)

# %%
df["sentiment"] = df["overall"].apply(lambda x: 1 if x > 3 else -1 if x < 3 else 0)
df.head()

# %%
df["sentiment"].value_counts()

# %%
df.drop("overall", axis=1, inplace=True)

# %%
df.head()

# %%

