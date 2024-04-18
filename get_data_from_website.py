
import time
import requests
import os
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract, extract_metadata

def getTextFrom_url(url):
    data={}
    html_doc = requests.get(url)
    soup = BeautifulSoup(html_doc.text, 'html.parser')
    data['url'] = url
    data['title'] = soup.title.string
    data['body'] = soup.get_text()
    return data
def get_Dataset(url):
    urls = sitemap_search(url)
    data = []
    for myurl in tqdm(urls, desc="URLs"):
        data.append(getTextFrom_url(myurl))
  
    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = get_Dataset("https://botpenguin.com/")
    df.to_csv("new_data.csv", index=False)