import json
import gzip
import pandas as pd

# Converting json.gz to dataframe.
# Code taken from the site the data is from
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

try:
    df = getDF('Musical_Instruments_5.json.gz')
    df2 = getDF('Arts_Crafts_and_Sewing_5.json.gz')
    df3 = getDF('All_Beauty_5.json.gz')
    df4 = getDF('AMAZON_FASHION_5.json.gz')
    df5 = getDF('Appliances_5.json.gz')
    df6 = getDF('Software_5.json.gz')
    frames = [df, df2, df3, df4, df5, df6]
    combined = pd.concat(frames)
    combined.to_csv('combined.csv', index=False)
except:
    print("Error loading data")
    print("Ensure the following files are present")
    print("Musical_Instruments_5.json.gz")
    print("Arts_Crafts_and_Sewing_5.json.gz")
    print("All_Beauty_5.json.gz")
    print("AMAZON_FASHION_5.json.gz")
    print("Appliances_5.json.gz")
    print("Software_5.json.gz")

