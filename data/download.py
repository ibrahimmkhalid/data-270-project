import wget
import os
import bs4
import pandas as pd

index_page = "https://cseweb.ucsd.edu//~jmcauley/datasets/amazon_v2/index.html"
index_page_file = wget.download(index_page)
index_page_content = open(index_page_file, "r").read()
index_page_soup = bs4.BeautifulSoup(index_page_content, "html.parser")

full_dataset = pd.DataFrame(columns=["name", "url", "size"])
small_dataset = pd.DataFrame(columns=["name", "url", "size"])

tables = index_page_soup.find_all("table", class_="code-table")

table = tables[0]
rows = table.find_all("tr")
for row in rows:
    cells = row.find_all("td")
    if len(cells) > 0:
        name = cells[0].text
        url = cells[1].find("a")["href"]
        dataset_size = cells[1].text.split(" ")[1:]
        dataset_size = " ".join(dataset_size)
        full_dataset.loc[len(full_dataset)] = [name, url, dataset_size]


table = tables[1]
rows = table.find_all("tr")
for row in rows:
    cells = row.find_all("td")
    if len(cells) > 0:
        name = cells[0].text
        url = cells[1].find("a")["href"]
        dataset_size = cells[1].text.split(" ")[1:]
        dataset_size = " ".join(dataset_size)
        small_dataset.loc[len(small_dataset)] = [name, url, dataset_size]

print()
print(f"To download the entire (34GB) dataset, visit {index_page} and download manually")
full = input("Do you want to a full dataset or a small 5-core dataset? (full/small): ")
if full == "full":
    dataset = full_dataset
else:
    dataset = small_dataset
print()
print(dataset[["name", "size"]])
print()
dataset_index = int(input("Enter the index of the dataset you want to download: "))
print()
dataset_name = dataset.iloc[dataset_index]["name"]
confirm = input(f"Do you want to download the {dataset_name} dataset? (yes/no): ")
print()
if confirm == "yes":
    dataset_url = dataset.iloc[dataset_index]["url"]
    dataset_file = wget.download(dataset_url)
    print(f"{dataset_name} dataset has been downloaded.")


os.remove(index_page_file)
