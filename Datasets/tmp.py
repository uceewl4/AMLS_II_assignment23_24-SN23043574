import json
import random
from urllib import request
import requests
from PIL import Image
from io import BytesIO
from urllib import request

with open("./Datasets/raw/validation.json", "r") as f:
    data = json.load(f)

# # print(data)
opener = request.build_opener()
# 构建请求头列表每次随机选择一个
ua_list = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36 Edg/103.0.1264.62",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
    "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.81 Safari/537.36 SE 2.X MetaSr 1.0",
]
opener.addheaders = [("User-Agent", random.choice(ua_list))]
request.install_opener(opener)
train_image = []
remove = []
for i in data["images"]:
    url = i["url"][0]
    print(url)
    path = f"./Datasets/raw/validation/{i['image_id']}.jpg"
    # img = request.urlretrieve(url, path)
    r = requests.get(url)
    # print(f"{i['image_id']}" + " " + f"{r.status_code}")
    # user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36"
    # headers = {"user-agent": user_agent}
    # req = request.Request(url=url, headers=headers)
    # response = request.urlopen(req, timeout=10000)
    # if response.getcode() == 200:
    #     print(i["image_id"])
    #     img = Image.open(BytesIO(response.read())).convert("RGB")
    #     img.save(path)
    #     train_image.append(i["image_id"])
    # else:
    #     remove.append(i["image_id"])

    if r.status_code == 200:
        print(i["image_id"])
        with open(path, "wb") as f:
            f.write(r.content)
    else:
        remove.append(i["image_id"])

print(len(train_image))

train_labels = []
for i in data["annotations"]:
    train_labels.append[i["label_id"]]
print(len(train_labels))
