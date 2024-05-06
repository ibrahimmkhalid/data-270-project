# DATA270

- `eda_*` - Files that perform exploratory data analysis
- `model_*` - Files implementing a specific model
- `data/download.py` - interactive data downloading tool from Amazon reviews dataset source
- `data/merge_datasets.py` - combine select .json.gz files to combined.csv
- `data/combined.csv` - combination of Fashion, Beauty, Appliances, Arts and crafts, Musical instruments, and Software product categories
- `models/` - pickle files for models, vectorizers
- `results/` - results of eda, testing, modelling, etc
- `requirements.txt` - installs all required libraries, except cuML and cuDF for Google Colab
---

# To run .pkl files:

- `model_logreg.pkl` - Training data was based on the baseline reviewTextSummary, transformed by a TfidfVectorizer(). Running the notebook should properly create the test data.
- `model_svm_clf.pkl` and `modle_svm_vectorizer.pkl` - Running the `model_svm_final.ipynb` notebook on **Google Colab** in order will run until the pickle file as long as it is found on the specified path
  - need to change the `work_path` to appropriate value

Data Analytics Processes Project. Jan 2024 - May 2024 
