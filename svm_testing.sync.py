import pandas as pd
import nltk
import re
import random
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# %%
baseline = "baseline"
stem = "stem"
lem = "lem"
bow = "bow"
tfidf = "tfidf"
random_state = 42
small_n = 2500
large_n = 5000
vlarge_n = 40000
random.seed(random_state)
data_path = "./data/combined.csv"

# %% [markdown]
# # EDA and simple preprocessing

# %%
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
df["reviewTextWithSummary"] = df["summary"] + " " + df["reviewText"]
df.drop(["overall", "summary"], axis=1, inplace=True)
df.head()

# %%
df["sentiment"].value_counts()

# %%
df.head()

# %% [markdown]
# # Model experimentation

# %%
df_balanced_small = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=small_n, random_state=random_state, replace=True))
    .reset_index(drop=True)
)
df_balanced_small["sentiment"].value_counts()

# %%
df_balanced_small.head()

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
    if type_proc == baseline:
        return word
    elif type_proc == stem:
        return PorterStemmer().stem(word)
    elif type_proc == lem:
        return WordNetLemmatizer().lemmatize(word)
    else:
        raise ValueError("Invalid Preprocessing Type")


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
    cols = x_train.columns

    textcol = "reviewText"
    if "reviewText" not in cols and "reviewTextWithSummary" in cols:
        textcol = "reviewTextWithSummary"
    x_train[textcol] = x_train[textcol].apply(
        lambda x: preprocess_text(x, STOP_WORDS, proc)
    )
    x_test[textcol] = x_test[textcol].apply(
        lambda x: preprocess_text(x, STOP_WORDS, proc)
    )
    return x_train, x_test


# %%
def pipeline(cols, test_size, proc, vectorizer, df=df, random_state=random_state):
    if "reviewText" not in cols and "reviewTextWithSummary" not in cols:
        raise ValueError("Must contain reviewText or reviewTextWithSummary")

    textcol = "reviewText"
    if "reviewText" not in cols and "reviewTextWithSummary" in cols:
        textcol = "reviewTextWithSummary"
    x_train, x_test, y_train, y_test = my_train_test_split(
        cols, test_size, df, random_state
    )
    x_train, x_test = apply_preprocessing(proc, x_train, x_test)
    if vectorizer == bow:
        vectorizer = CountVectorizer()
    elif vectorizer == tfidf:
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid Vectorizer")
    x_train = vectorizer.fit_transform(x_train[textcol])
    x_test = vectorizer.transform(x_test[textcol])
    return x_train, x_test, y_train, y_test

# %% [markdown]
# ## Small Balanced Dataset

# %%
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf", "linear"],
}
n_jobs = -1
verbose = 0
cv = 3

# %%
compare_list = pd.DataFrame(columns=["Params", "Config", "Accuracy Score"])

# %%
code_gen = False
col_comb = [
    ["reviewText"],
    ["reviewText", "verified"],
    ["reviewTextWithSummary"],
    ["reviewTextWithSummary", "verified"],
]
proc_comb = [None, baseline, stem, lem]
vectorizer_comb = [bow, tfidf]
if code_gen:
    for col in col_comb:
        for proc in proc_comb:
            for vectorizer in vectorizer_comb:
                params = {"col": col, "test_size": 0.25, "proc": proc, "vectorizer": vectorizer}
                print(f"""
# %%
x_train, x_test, y_train, y_test = pipeline({col}, 0.25, {proc}, {vectorizer}, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {params}, accuracy]
""")


# %%
# below is code genderated by above cell, to make changes to the code, edit the
# above cell and run it, pasting its contents between the markers
# %%
####### START OF GENERATED CODE #######
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]
#######  END OF GENERATED CODE  #######

# %%
compare_list = compare_list.sort_values(
    by="Accuracy Score", ascending=False
).reset_index(drop=True)
display(compare_list)

# %%
print("Best Configuration on small balnaced dataset")
print("Score :: ", compare_list.loc[0]["Accuracy Score"])
print("SVC   :: ", compare_list.loc[0]["Params"])
print("data  :: ", compare_list.loc[0]["Config"])
# %%
compare_list.to_csv("./results/svm_compare_list_small_balanced.csv", index=False)

# %% [markdown]
# - Across all tests, reviewText with summary performed better than without summary.
# - Whether the verified column was included or not did not have any significant impact on the accuracy.
# - The RBF kernel performed better than the linear kernel in almost all cases.
# - Models performed better when the C value was = 1

# %%
df_balanced_vlarge = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=vlarge_n, random_state=random_state, replace=True))
    .reset_index(drop=True)
)
svm_compare_list = pd.DataFrame(columns=["Model", "Accuracy Score", "Precision", "Recall", "F1Score"])

# %%
svc_balanced_small = SVC(C=1, gamma=1, kernel="rbf")
x_train, x_test, y_train, y_test = pipeline(
    ["reviewTextWithSummary"], 0.25, None, tfidf, df_balanced_vlarge
)
svc_balanced_small.fit(x_train, y_train)
y_pred = svc_balanced_small.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
svm_compare_list.loc[len(svm_compare_list)] = ["SVC Balanced Small", accuracy, precision, recall, f1]

# %% [markdown]
# ## Large Balanced Dataset

# %%
df_balanced_large = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=large_n, random_state=random_state, replace=True))
    .reset_index(drop=True)
)


# %%
param_grid = {
    "C": [0.1, 0.5, 1, 5, 10],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf"],
}
n_jobs = -1
verbose = 0
cv = 3

# %%
compare_list = pd.DataFrame(columns=["Params", "Config", "Accuracy Score"])

# %%
code_gen = False
proc_comb = [None, baseline, stem, lem]
vectorizer_comb = [bow, tfidf]
col = ["reviewTextWithSummary"]
if code_gen:
    for proc in proc_comb:
        for vectorizer in vectorizer_comb:
            params = {"col": col, "test_size": 0.25, "proc": proc, "vectorizer": vectorizer}
            print(f"""
# %%
x_train, x_test, y_train, y_test = pipeline({col}, 0.25, {proc}, {vectorizer}, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {params}, accuracy]
""")

# %%
# below is code genderated by above cell, to make changes to the code, edit the
# above cell and run it, pasting its contents between the markers
# %%
####### START OF GENERATED CODE #######
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, bow, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, tfidf, df_balanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]
#######  END OF GENERATED CODE  #######
# %%
compare_list = compare_list.sort_values(
    by="Accuracy Score", ascending=False
).reset_index(drop=True)
display(compare_list)

# %%
print("Best Configuration on larger dataset")
print("Score :: ", compare_list.loc[0]["Accuracy Score"])
print("SVC   :: ", compare_list.loc[0]["Params"])
print("data  :: ", compare_list.loc[0]["Config"])

# %%
compare_list.to_csv("./results/svm_compare_list_large_balanced.csv", index=False)

# %% [markdown]
# ...
# %%
svc_balanced_large = SVC(C=1, gamma=1, kernel="rbf")
x_train, x_test, y_train, y_test = pipeline(
    ["reviewTextWithSummary"], 0.25, None, tfidf, df_balanced_vlarge
)
svc_balanced_large.fit(x_train, y_train)
y_pred = svc_balanced_large.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
svm_compare_list.loc[len(svm_compare_list)] = ["SVC Balanced Large", accuracy, precision, recall, f1]

# %%
# 3 times the small_n to create an unbalanced dataset of the same size as the balanced dataset
df_unbalanced_small = df.sample(n=3*small_n, random_state=random_state, replace=True)
df_unbalanced_small["sentiment"].value_counts()

# %% [markdown]
# ## Small Unbalanced Dataset

# %%
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf", "linear"],
}
n_jobs = -1
verbose = 0
cv = 3

# %%
compare_list = pd.DataFrame(columns=["Params", "Config", "Accuracy Score"])

# %%
code_gen = False
col_comb = [
    ["reviewText"],
    ["reviewText", "verified"],
    ["reviewTextWithSummary"],
    ["reviewTextWithSummary", "verified"],
]
proc_comb = [None, baseline, stem, lem]
vectorizer_comb = [bow, tfidf]
if code_gen:
    for col in col_comb:
        for proc in proc_comb:
            for vectorizer in vectorizer_comb:
                params = {"col": col, "test_size": 0.25, "proc": proc, "vectorizer": vectorizer}
                print(f"""
# %%
x_train, x_test, y_train, y_test = pipeline({col}, 0.25, {proc}, {vectorizer}, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {params}, accuracy]
""")

# %%
# below is code genderated by above cell, to make changes to the code, edit the
# above cell and run it, pasting its contents between the markers
# %%
####### START OF GENERATED CODE #######
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, bow, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, tfidf, df_unbalanced_small)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]
#######  END OF GENERATED CODE  #######
# %%
compare_list = compare_list.sort_values(
    by="Accuracy Score", ascending=False
).reset_index(drop=True)
display(compare_list)

# %%
compare_list.to_csv("./results/svm_compare_list_small_unbalanced.csv", index=False)

# %%
print("Best Configuration on small unbalanced dataset")
print("Score :: ", compare_list.loc[0]["Accuracy Score"])
print("SVC   :: ", compare_list.loc[0]["Params"])
print("data  :: ", compare_list.loc[0]["Config"])

# %% [markdown]
# ...

# %%
df_unbalanced_vlarge = df.sample(n=3*vlarge_n, random_state=random_state, replace=True)

# %%
svc_unbalanced_small = SVC(C=10, gamma=0.01, kernel="rbf")
x_train, x_test, y_train, y_test = pipeline(
    ["reviewTextWithSummary"], 0.25, None, bow, df_unbalanced_vlarge
)
svc_unbalanced_small.fit(x_train, y_train)
y_pred = svc_unbalanced_small.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
svm_compare_list.loc[len(svm_compare_list)] = ["SVC unbalanced Small", accuracy, precision, recall, f1]


# %% [markdown]
# ## Large Unbalanced Dataset

# %%
df_unbalanced_large = df.sample(n=3*large_n, random_state=random_state, replace=True)
df_unbalanced_large["sentiment"].value_counts()

# %%
param_grid = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
    "kernel": ["rbf"],
}

n_jobs = -1
verbose = 0
cv = 3

# %%
compare_list = pd.DataFrame(columns=["Params", "Config", "Accuracy Score"])

# %%
code_gen = False
proc_comb = [None, baseline, stem, lem]
vectorizer_comb = [bow, tfidf]
col = ["reviewTextWithSummary"]
if code_gen:
    for proc in proc_comb:
        for vectorizer in vectorizer_comb:
            params = {"col": col, "test_size": 0.25, "proc": proc, "vectorizer": vectorizer}
            print(f"""
# %%
x_train, x_test, y_train, y_test = pipeline({col}, 0.25, {proc}, {vectorizer}, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {params}, accuracy]
""")

# %%
# below is code genderated by above cell, to make changes to the code, edit the
# above cell and run it, pasting its contents between the markers

# %%
####### START OF GENERATED CODE #######
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, bow, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, tfidf, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, bow, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, tfidf, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, bow, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, tfidf, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, bow, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, tfidf, df_unbalanced_large)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]
#######  END OF GENERATED CODE  #######

# %%
compare_list = compare_list.sort_values(
    by="Accuracy Score", ascending=False
).reset_index(drop=True)
display(compare_list)

# %%
compare_list.to_csv("./results/svm_compare_list_large_unbalanced.csv", index=False)

# %%
print("Best Configuration on larger dataset")
print("Score :: ", compare_list.loc[0]["Accuracy Score"])
print("SVC   :: ", compare_list.loc[0]["Params"])
print("data  :: ", compare_list.loc[0]["Config"])

# %%
svc_unbalanced_large = SVC(C=10, gamma=0.1, kernel="rbf")
x_train, x_test, y_train, y_test = pipeline(
    ["reviewTextWithSummary"], 0.25, None, tfidf, df_unbalanced_small
)
svc_unbalanced_large.fit(x_train, y_train)
y_pred = svc_unbalanced_large.predict(x_test)
print(classification_report(y_test, y_pred))

# %% [markdown]
# ## Comparing the best models from each experiment on the full dataset
