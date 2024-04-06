import pandas as pd
import nltk
import re
import random
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from scipy.sparse import hstack

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
testing_n = 50
large_n = 100
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
df_testing = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=testing_n, random_state=random_state, replace=True))
    .reset_index(drop=True)
)
df_testing["sentiment"].value_counts()

# %%
df_testing.head()

# %%
print("Dataset size:", len(df_testing))

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
def add_col(x, col):
    col = np.array([col]).T
    return hstack([x, col])


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
    x_train_ = vectorizer.fit_transform(x_train[textcol])
    x_test_ = vectorizer.transform(x_test[textcol])

    if "verified" in cols:
        x_train = add_col(x_train_, x_train["verified"])
        x_test = add_col(x_test_, x_test["verified"])
    else:
        x_train = x_train_
        x_test = x_test_
    return x_train, x_test, y_train, y_test


# %% [markdown]
# ## Testing different configs

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
                params = {
                    "col": col,
                    "test_size": 0.25,
                    "proc": proc,
                    "vectorizer": vectorizer,
                }
                print(
                    f"""
# %%
x_train, x_test, y_train, y_test = pipeline({col}, 0.25, {proc}, {vectorizer}, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {params}, accuracy]
"""
                )


# %%
# below is code genderated by above cell, to make changes to the code, edit the
# above cell and run it, pasting its contents between the markers
# %%
####### START OF GENERATED CODE #######
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': None, 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'baseline', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'stem', 'vectorizer': 'tfidf'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
accuracy = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'test_size': 0.25, 'proc': 'lem', 'vectorizer': 'bow'}, accuracy]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, tfidf, df_testing)
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
print(f"Best Configuration on testing dataset (size={len(df_testing)}):")
print("Score :: ", compare_list.loc[0]["Accuracy Score"])
print("SVC   :: ", compare_list.loc[0]["Params"])
print("data  :: ", compare_list.loc[0]["Config"])
# %%
compare_list.to_csv("./results/svm_compare_list.csv", index=False)

# %% [markdown]
# - Across all tests, reviewText with summary performed better than reviewText without summary.
# - The RBF kernel performed better than the linear kernel in almost all cases.
# - The top configuration was as follows:
#   - Data::
#     - Columns used: reviewTextWithSummary
#     - Text preprocessing step: None
#     - Text vectorizer: tfidf
#   - SVC::
#     - C=1
#     - gamma=1
#     - kernel=rbf
#
# Using these parameters, lets build a model on a larger dataset.


# %% [markdown]
# ## Building larger model

# %%
df_large = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=large_n, random_state=random_state, replace=True))
    .reset_index(drop=True)
)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    df_large[["reviewTextWithSummary"]],
    df_large["sentiment"],
    test_size=0.25,
    random_state=random_state,
)

# %%
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train["reviewTextWithSummary"])
x_test = vectorizer.transform(x_test["reviewTextWithSummary"])

# %%
svc_testing_df_large = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_testing_df_large.fit(x_train, y_train)

# %%
y_pred = svc_testing_df_large.predict(x_test)

# %%
sample = [
    "I loved this product, it was amazing",
    "I hated this product, it was terrible",
    "This product was okay, it was fine",
]

# %%
sample_ = [vectorizer.transform([x]) for x in sample]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_testing_df_large.predict(p))
    print()

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# - Results seem very good with an F1 Score of 0.87
# - Interestingly, the top perfomring model did not use any text preprocessing
# - The next best performing models that used different text preprocessing were as follows:
#   - baseline text preprocessing with tfidf vectorizer: score of 0.7728
#   - lemmatized text preprocessing with tfidf vectorizer: score of 0.7677
#   - stemmed text preprocessing with tfidf vectorizer: score of 0.7642
# - These results are within 2% than the selected model, so it may be worth exploring these models further

# %% [markdown]
# ### Baseline with tfidf

# %%
x_train, x_test, y_train, y_test = train_test_split(
    df_large[["reviewTextWithSummary"]],
    df_large["sentiment"],
    test_size=0.25,
    random_state=random_state,
)

# %%
x_train["reviewTextWithSummary"] = x_train["reviewTextWithSummary"].apply(
    lambda x: preprocess_text(x, STOP_WORDS, baseline)
)
x_test["reviewTextWithSummary"] = x_test["reviewTextWithSummary"].apply(
    lambda x: preprocess_text(x, STOP_WORDS, baseline)
)

# %%
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train["reviewTextWithSummary"])
x_test = vectorizer.transform(x_test["reviewTextWithSummary"])

# %%
svc_baseline_tfidf = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_baseline_tfidf.fit(x_train, y_train)

# %%
y_pred = svc_baseline_tfidf.predict(x_test)

# %%
sample = [
    "I loved this product, it was amazing",
    "I hated this product, it was terrible",
    "This product was okay, it was fine",
]

# %%
sample_ = [preprocess_text(x, STOP_WORDS, baseline) for x in sample]

# %%
sample_ = [vectorizer.transform([x]) for x in sample_]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_baseline_tfidf.predict(p))
    print()


# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
#

# %% [markdown]
# ### Lemmatized with tfidf
# %%
x_train, x_test, y_train, y_test = train_test_split(
    df_large[["reviewTextWithSummary"]],
    df_large["sentiment"],
    test_size=0.25,
    random_state=random_state,
)

# %%
x_train["reviewTextWithSummary"] = x_train["reviewTextWithSummary"].apply(
    lambda x: preprocess_text(x, STOP_WORDS, lem)
)
x_test["reviewTextWithSummary"] = x_test["reviewTextWithSummary"].apply(
    lambda x: preprocess_text(x, STOP_WORDS, lem)
)

# %%
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train["reviewTextWithSummary"])
x_test = vectorizer.transform(x_test["reviewTextWithSummary"])

# %%
svc_lematized_tfidf = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_lematized_tfidf.fit(x_train, y_train)

# %%
y_pred = svc_lematized_tfidf.predict(x_test)

# %%
sample = [
    "I loved this product, it was amazing",
    "I hated this product, it was terrible",
    "This product was okay, it was fine",
]

# %%
sample_ = [preprocess_text(x, STOP_WORDS, lem) for x in sample]

# %%
sample_ = [vectorizer.transform([x]) for x in sample_]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_lematized_tfidf.predict(p))
    print()

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
#

# %% [markdown]
# ### Stemmed with tfidf
# %%
x_train, x_test, y_train, y_test = train_test_split(
    df_large[["reviewTextWithSummary"]],
    df_large["sentiment"],
    test_size=0.25,
    random_state=random_state,
)

# %%
x_train["reviewTextWithSummary"] = x_train["reviewTextWithSummary"].apply(
    lambda x: preprocess_text(x, STOP_WORDS, stem)
)
x_test["reviewTextWithSummary"] = x_test["reviewTextWithSummary"].apply(
    lambda x: preprocess_text(x, STOP_WORDS, stem)
)

# %%
vectorizer = TfidfVectorizer()
x_train = vectorizer.fit_transform(x_train["reviewTextWithSummary"])
x_test = vectorizer.transform(x_test["reviewTextWithSummary"])

# %%
svc_stemmed_tfidf = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_stemmed_tfidf.fit(x_train, y_train)

# %%
y_pred = svc_stemmed_tfidf.predict(x_test)

# %%
sample = [
    "I loved this product, it was amazing",
    "I hated this product, it was terrible",
    "This product was okay, it was fine",
]

# %%
sample_ = [preprocess_text(x, STOP_WORDS, stem) for x in sample]

# %%
sample_ = [vectorizer.transform([x]) for x in sample_]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_stemmed_tfidf.predict(p))
    print()

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
#

# %% [markdown]
#
