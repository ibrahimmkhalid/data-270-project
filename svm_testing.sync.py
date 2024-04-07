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
from sklearn.metrics import f1_score
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
testing_n = 5000
large_n = 50000
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
compare_list = pd.DataFrame(
    columns=["Grid Params", "Data config and preprocessing", "Grid Score"]
)

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
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {params}, grid_score]
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
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': None, 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': None, 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': 'baseline', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': 'baseline', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': 'stem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': 'stem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': 'lem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText'], 'proc': 'lem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': None, 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': None, 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': 'baseline', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': 'baseline', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': 'stem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': 'stem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': 'lem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewText', 'verified'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewText', 'verified'], 'proc': 'lem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': None, 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': None, 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': 'baseline', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': 'baseline', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': 'stem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': 'stem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': 'lem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary'], 'proc': 'lem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': None, 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, None, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': None, 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': 'baseline', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, baseline, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': 'baseline', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': 'stem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, stem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': 'stem', 'vectorizer': 'tfidf'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, bow, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': 'lem', 'vectorizer': 'bow'}, grid_score]


# %%
x_train, x_test, y_train, y_test = pipeline(['reviewTextWithSummary', 'verified'], 0.25, lem, tfidf, df_testing)
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=verbose, n_jobs=n_jobs, cv=cv)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
print(classification_report(y_test, y_pred))
grid_score = grid.score(x_test, y_test)
compare_list.loc[len(compare_list)] = [grid.best_params_, {'col': ['reviewTextWithSummary', 'verified'], 'proc': 'lem', 'vectorizer': 'tfidf'}, grid_score]
#######  END OF GENERATED CODE  #######

# %%
pd.set_option("display.max_colwidth", None)
compare_list = compare_list.sort_values(by="Grid Score", ascending=False).reset_index(drop=True)
display(compare_list)

# %%
print(f"Best Configuration on testing dataset (size={len(df_testing)}):")
print("Score :: ", compare_list.loc[0]["Grid Score"])
print("SVC   :: ", compare_list.loc[0]["Grid Params"])
print("data  :: ", compare_list.loc[0]["Data config and preprocessing"])

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
sample = [
    "I loved this product, it was amazing",
    "I hated this product, it was terrible",
    "This product was okay, it was fine",
    "I am not sure how I feel about this product",
    "Apple really outdid themselves with this product",
    "The engine was really loud, but otherwise the car was fine",
]

# %% [markdown]
# ### Top performing model

# %%
x_train, x_test, y_train, y_test = train_test_split(
    df_large[["reviewTextWithSummary"]],
    df_large["sentiment"],
    test_size=0.25,
    random_state=random_state,
)

# %%
vec_testing_top_config = TfidfVectorizer()
x_train = vec_testing_top_config.fit_transform(x_train["reviewTextWithSummary"])
x_test = vec_testing_top_config.transform(x_test["reviewTextWithSummary"])

# %%
svc_testing_top_config = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_testing_top_config.fit(x_train, y_train)

# %%
y_pred = svc_testing_top_config.predict(x_test)

# %%
sample_ = [vec_testing_top_config.transform([x]) for x in sample]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_testing_top_config.predict(p))
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
print("Best configuration using baseline text preprocessing")
print("Score :: ", compare_list.loc[4]["Grid Score"])
print("SVC   :: ", compare_list.loc[4]["Grid Params"])
print("data  :: ", compare_list.loc[4]["Data config and preprocessing"])

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
vec_baseline_tfidf = TfidfVectorizer()
x_train = vec_baseline_tfidf.fit_transform(x_train["reviewTextWithSummary"])
x_test = vec_baseline_tfidf.transform(x_test["reviewTextWithSummary"])

# %%
svc_baseline_tfidf = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_baseline_tfidf.fit(x_train, y_train)

# %%
y_pred = svc_baseline_tfidf.predict(x_test)

# %%
sample_ = [preprocess_text(x, STOP_WORDS, baseline) for x in sample]
sample_ = [vec_baseline_tfidf.transform([x]) for x in sample_]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_baseline_tfidf.predict(p))
    print()


# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Lemmatized with tfidf

# %%
print("Best configuration using baseline text preprocessing")
print("Score :: ", compare_list.loc[6]["Grid Score"])
print("SVC   :: ", compare_list.loc[6]["Grid Params"])
print("data  :: ", compare_list.loc[6]["Data config and preprocessing"])

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
vec_lematized_tfidf = TfidfVectorizer()
x_train = vec_lematized_tfidf.fit_transform(x_train["reviewTextWithSummary"])
x_test = vec_lematized_tfidf.transform(x_test["reviewTextWithSummary"])

# %%
svc_lematized_tfidf = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_lematized_tfidf.fit(x_train, y_train)

# %%
y_pred = svc_lematized_tfidf.predict(x_test)

# %%
sample_ = [preprocess_text(x, STOP_WORDS, lem) for x in sample]
sample_ = [vec_lematized_tfidf.transform([x]) for x in sample_]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_lematized_tfidf.predict(p))
    print()

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# ### Stemmed with tfidf

# %%
print("Best configuration using baseline text preprocessing")
print("Score :: ", compare_list.loc[8]["Grid Score"])
print("SVC   :: ", compare_list.loc[8]["Grid Params"])
print("data  :: ", compare_list.loc[8]["Data config and preprocessing"])

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
vec_stemmed_tfidf = TfidfVectorizer()
x_train = vec_stemmed_tfidf.fit_transform(x_train["reviewTextWithSummary"])
x_test = vec_stemmed_tfidf.transform(x_test["reviewTextWithSummary"])

# %%
svc_stemmed_tfidf = SVC(C=1, gamma=1, kernel="rbf")

# %%
svc_stemmed_tfidf.fit(x_train, y_train)

# %%
y_pred = svc_stemmed_tfidf.predict(x_test)

# %%
sample_ = [preprocess_text(x, STOP_WORDS, stem) for x in sample]
sample_ = [vec_stemmed_tfidf.transform([x]) for x in sample_]

# %%
for s, p in zip(sample, sample_):
    print(s)
    print(svc_stemmed_tfidf.predict(p))
    print()

# %%
print(classification_report(y_test, y_pred))

# %% [markdown]
# - All models perform within a percent of each other based on their own test sets
# - lets compare each of these models on the full dataset

# %% [markdown]
# ## Testing on full dataset

# %%
x = df[["reviewTextWithSummary"]]
y = df["sentiment"]

# %%
x_top_config = vec_testing_top_config.transform(x["reviewTextWithSummary"])
x_baseline_tfidf = vec_baseline_tfidf.transform(
    x["reviewTextWithSummary"].apply(lambda x: preprocess_text(x, STOP_WORDS, baseline))
)
x_lematized_tfidf = vec_lematized_tfidf.transform(
    x["reviewTextWithSummary"].apply(lambda x: preprocess_text(x, STOP_WORDS, lem))
)
x_stemmed_tfidf = vec_stemmed_tfidf.transform(
    x["reviewTextWithSummary"].apply(lambda x: preprocess_text(x, STOP_WORDS, stem))
)

# %%
y_pred_top_config = svc_testing_top_config.predict(x_top_config)
y_pred_baseline_tfidf = svc_baseline_tfidf.predict(x_baseline_tfidf)
y_pred_lematized_tfidf = svc_lematized_tfidf.predict(x_lematized_tfidf)
y_pred_stemmed_tfidf = svc_stemmed_tfidf.predict(x_stemmed_tfidf)

# %%
print("Top Config")
print(classification_report(y, y_pred_top_config))
score_top_config = f1_score(y, y_pred_top_config, average="weighted")


# %%
print("Baseline with tfidf")
print(classification_report(y, y_pred_baseline_tfidf))
score_baseline_tfidf = f1_score(y, y_pred_baseline_tfidf, average="weighted")


# %%
print("Lematized with tfidf")
print(classification_report(y, y_pred_lematized_tfidf))
score_lematized_tfidf = f1_score(y, y_pred_lematized_tfidf, average="weighted")


# %%
print("Stemmed with tfidf")
print(classification_report(y, y_pred_stemmed_tfidf))
score_stemmed_tfidf = f1_score(y, y_pred_stemmed_tfidf, average="weighted")


# %%
print("Scores")
print("Overall top config (No preprocessing) :: ", score_top_config)
print("Baseline preprocessing top config     :: ", score_baseline_tfidf)
print("Lemmatized preprocessing top config   :: ", score_lematized_tfidf)
print("Stemmed preprocessing top config      :: ", score_stemmed_tfidf)

# %% [markdown]
# ## Conclusion
# - Performance ranking is the same as the experimental results from before
# - All selected models perform at a very high level, all around 88% accurate with F1 scores around 0.9
# - The overall top performing model was the one that did not use any text preprocessing with the following:
#   - Text was not preprocessesed
#   - Text was vectorized using the tfidf vectorizer
#   - SVC with C=1, gamma=1, kernel=rbf
#   - Weighted F1 score = 0.9067

