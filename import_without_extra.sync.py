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
import pandas as pd
import json
import gzip


def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path, stop=0):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if stop > 0 and i == stop:
            break
    return pd.DataFrame.from_dict(df, orient="index")

# %%
df = pd.DataFrame(columns=["overall", "verified", "reviewText", "summary", "sentiment"])

# %%
peek = 1_000_000
stop = 1000
i = 0

for review in parse("./data/All_Amazon_Review.json.gz"):
    i += 1
    if stop > 0 and stop == i:
        break
    if i % peek == 0:
        print(f"Processed {i / peek:.2f} million reviews")

    if "overall" not in review or "verified" not in review or "reviewText" not in review or "summary" not in review:
        continue
    df.loc[i] = [review["overall"], review["verified"], review["reviewText"], review["summary"], 0]



# %%
df.head()

# %%
df.shape

# %%
df_size =  df.memory_usage(index=True).sum()
print("size of df in bytes: ", df_size)

# %%
df1 = getDF("./data/All_Amazon_Review.json.gz", stop)

# %%
df1.head()

# %%
df1.shape

# %%
df1_size =  df1.memory_usage(index=True).sum()
print("size of df1 in bytes: ", df1_size)

# %%
print("ratio of reading the whole file vs parsing the required columns: ", df1_size / df_size)
