# import warnings
# warnings.filterwarnings("ignore")

import pandas as pd
import nltk
import re
import random
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

random_state = 42
random.seed(random_state)
from nltk.corpus import stopwords

nltk.download("stopwords")

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
df["reviewText"] = df["summary"] + " " + df["reviewText"]
df.drop(["overall", "summary"], axis=1, inplace=True)
df.head()

# %%
df["sentiment"].value_counts()

# %%
df.head()

# %%
df = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=100000, random_state=random_state, replace=True))
    .reset_index(drop=True)
)
df["sentiment"].value_counts()

# %%
df.head()

# %%
STOP_WORDS = set(stopwords.words("english"))


# %%
def preprocess_text(sentence, stop, type_proc=None):
    words = []
    for word in sentence.lower().strip().split():

        word = re.sub("\d", "", word)
        word = re.sub("[^\w\s]", "", word)

        if word not in stop and word != "":
            words.append(preprocess_type(word, type_proc))

    return " ".join(words)


# %%
def preprocess_type(word, type_proc):
    match type_proc:
        case "word":
            return word
        case "stem":
            return PorterStemmer().stem(word)
        case "lem":
            return WordNetLemmatizer().lemmatize(word)


# %%
def my_train_test_split(cols, test_size, df=df, random_state=random_state):
    x_train, x_test, y_train, y_test = train_test_split(
        df[cols], df["sentiment"], test_size=test_size, random_state=random_state
    )
    return x_train, x_test, y_train, y_test


# %%
def apply_preprocessing(proc, x_train, x_test):
    if proc is None:
        return x_train, x_test
    x_train["reviewText"] = x_train["reviewText"].apply(
        lambda x: preprocess_text(x, STOP_WORDS, proc)
    )
    x_test["reviewText"] = x_test["reviewText"].apply(
        lambda x: preprocess_text(x, STOP_WORDS, proc)
    )
    return x_train, x_test


# %%
def pipeline(cols, test_size, proc, vectorizer, df=df, random_state=random_state):
    assert "reviewText" in cols
    x_train, x_test, y_train, y_test = my_train_test_split(
        cols, test_size, df, random_state
    )
    x_train, x_test = apply_preprocessing(proc, x_train, x_test)
    if vectorizer == "bow":
        vectorizer = CountVectorizer()
    elif vectorizer == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid Vectorizer")
    x_train = vectorizer.fit_transform(x_train["reviewText"])
    x_test = vectorizer.transform(x_test["reviewText"])
    return x_train, x_test, y_train, y_test


# %%
cols = ["reviewText"]
test_size = 0.2
proc = None
vectorizer = "bow"
x_train, x_test, y_train, y_test = pipeline(cols, test_size, proc, vectorizer)
