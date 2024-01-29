import sys
import os
import json
import argparse
from pandas import json_normalize
import urllib
from urllib import request
import time
from concurrent.futures import ThreadPoolExecutor


def download(name):
    def download_image(image_id, url):
        for i in range(100):
            try:
                path = f"./Datasets/raw/{name}/{image_id}.jpg"
                request.urlretrieve(url, path)
                break
            except:
                time.sleep(0.1)

    with open(f"./Datasets/raw/{name}.json", "r") as f:
        data = json.load(f)

    images = json_normalize(data["images"])
    images["url"] = images["url"].apply(lambda x: x[0])
    annotations = json_normalize(data["annotations"])
    total = images.merge(annotations, on="image_id")
    with ThreadPoolExecutor(max_workers=128) as executor:
        for index, row in total.iterrows():
            executor.submit(download_image, row["image_id"], row["url"])


download("train")
download("test")
download("validation")
