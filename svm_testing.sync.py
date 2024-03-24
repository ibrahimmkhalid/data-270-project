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
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
import re


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
df = pd.DataFrame(columns=["verified", "text", "sentiment"])

# %%
peek = 1_000
stop = peek * 10
i = 0

for review in parse("./data/All_Amazon_Review.json.gz"):
    i += 1
    if stop > 0 and stop == i:
        break
    if i % peek == 0:
        print(f"Processed {i / peek:.2f} million reviews")

    if "overall" not in review or "verified" not in review or "reviewText" not in review or "summary" not in review:
        continue
    text = review["summary"] + " " + review["reviewText"]
    text = re.sub(r"[^a-zA-Z\s]", "", text) # remove non-alphabetic characters
    text = text.lower()
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])


    df.loc[i] = [
        review["verified"],
        text,
        "positive" if review["overall"] > 3 else "negative" if review["overall"] < 3 else "neutral",
    ]


# %%
df.head()
