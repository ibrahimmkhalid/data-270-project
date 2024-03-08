# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


# %%
bad_reviews = []
ratings = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "other": 0}

no_reviewText = []
no_summary = []
number_of_nulls = 0
count_verified = 0

peek = 1_000_000
i = 0
stop = peek * 0

for review in parse("./data/All_Amazon_Review.json.gz"):
    i += 1
    if stop > 0 and stop == i:
        break
    if i % peek == 0:
        print(f"Processed {i / peek:.2f} million reviews")

    try:
        if review["verified"] == True:
            count_verified += 1
    except KeyError:
        bad_reviews.append(review)

    try:
        if review["reviewText"] is None:
            number_of_nulls += 1
            no_reviewText.append(review)
    except KeyError:
        number_of_nulls += 1
        no_reviewText.append(review)

    try:
        if review["summary"] is None:
            number_of_nulls += 1
            no_summary.append(review)
    except KeyError:
        number_of_nulls += 1
        no_summary.append(review)

    try:
        ratings[review["overall"]] += 1
    except:
        ratings["other"] += 1
        bad_reviews.append(review)

# %%
print(ratings)

# %%
max_i = sum(ratings.values())
print("Percentage of n-star reviews")
for k, v in ratings.items():
    print(f"{k}-star: {v / max_i * 100:.2f}%")

# %%
print(f"Number of null reviews: {number_of_nulls} (either missing reviewText or summary)")

# %%
for review in no_summary:
    print(review)
    print("\n\n")

# %%
for review in no_reviewText:
    print(review)
    print("\n\n")

# %%
print(f"Percentage of verified reviews: {count_verified / i * 100:.2f}%")

# %%
for review in bad_reviews:
    print(review)
    print("\n\n")
