# -*- encoding: utf-8 -*-
"""
@File    :   get_datatset.py
@Time    :   2024/02/24 21:20:53
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for getting raw dataset from SOP dataset as train/validation/test.
Notice that this file should not be run, it's used for split dataset into train/val/test with the whole original datasets
"""

# here put the import lib

import random
import os

# # split the whole SOP into 3:1:1 for train/val/test
# info = {}
# raw_path = "datasets/raw"
# # raw_data:{total_count:, class types:, num of each class:},
# # train:{total_count:, num of each class}
# info["raw_data"] = {}
# info["raw_data"]["class"] = len(os.listdir(raw_path))
# info["raw_data"]["total_count"] = 0
# info["train"] = {}
# info["train"]["total_count"] = 0
# info["val"] = {}
# info["val"]["total_count"] = 0
# info["test"] = {}
# info["test"]["total_count"] = 0

# for folder in os.listdir(raw_path):
#     folder_path = os.path.join(raw_path, folder)
#     length = len(os.listdir(folder_path))
#     info["raw_data"][f"{folder[:-6]}"] = length
#     info["raw_data"]["total_count"] += length

#     # train
#     train_index = random.sample([i for i in range(length)], int(3 * length / 5))
#     files = os.listdir(folder_path)
#     if not os.path.exists("datasets/raw/train"):
#         os.makedirs("datasets/raw/train")
#     for index, i in enumerate(train_index):
#         os.rename(
#             os.path.join(folder_path, files[i]),
#             f"datasets/raw/train/train_{folder[:-6]}_{index}.JPG",
#         )
#     info["train"][f"{folder[:-6]}"] = len(train_index)
#     info["train"]["total_count"] += len(train_index)

#     # validation
#     length = len(os.listdir(folder_path))
#     val_index = random.sample([i for i in range(length)], int(length / 2))
#     files = os.listdir(folder_path)
#     if not os.path.exists("datasets/raw/val"):
#         os.makedirs("datasets/raw/val")
#     for index, i in enumerate(val_index):
#         os.rename(
#             os.path.join(folder_path, files[i]),
#             f"datasets/raw/val/val_{folder[:-6]}_{index}.JPG",
#         )
#     info["val"][f"{folder[:-6]}"] = len(val_index)
#     info["val"]["total_count"] += len(val_index)

#     # test
#     length = len(os.listdir(folder_path))
#     files = os.listdir(folder_path)
#     if not os.path.exists("datasets/raw/test"):
#         os.makedirs("datasets/raw/test")
#     for index, file in enumerate(os.listdir(folder_path)):
#         os.rename(
#             os.path.join(folder_path, file),
#             f"datasets/raw/test/test_{folder[:-6]}_{index}.JPG",
#         )
#     info["test"][f"{folder[:-6]}"] = length
#     info["test"]["total_count"] += length

# print(info)


# considering large amount of datasets, just use 10000 images is more reasonable for most of the project
# train 6000:2400:2400 approximate 3:1:1
# to ensure balanced class at the same time, /12
info = {}
raw_path = "datasets/raw"
# raw_data:{total_count:, class types:, num of each class:},
# train:{total_count:, num of each class}
info["raw_data"] = {}
info["raw_data"]["class"] = len(os.listdir(raw_path))
info["raw_data"]["total_count"] = 0
info["train"] = {}
info["train"]["total_count"] = 0
info["val"] = {}
info["val"]["total_count"] = 0
info["test"] = {}
info["test"]["total_count"] = 0

for folder in os.listdir(raw_path):
    folder_path = os.path.join(raw_path, folder)
    length = len(os.listdir(folder_path))
    info["raw_data"][f"{folder[:-6]}"] = length
    info["raw_data"]["total_count"] += length

    # train
    train_index = random.sample([i for i in range(length)], 500)  # 6000/12
    files = os.listdir(folder_path)
    if not os.path.exists("datasets/raw/train"):
        os.makedirs("datasets/raw/train")
    for index, i in enumerate(train_index):
        os.rename(
            os.path.join(folder_path, files[i]),
            f"datasets/raw/train/train_{folder[:-6]}_{index}.JPG",
        )
    info["train"][f"{folder[:-6]}"] = len(train_index)
    info["train"]["total_count"] += len(train_index)

    # validation
    length = len(os.listdir(folder_path))
    val_index = random.sample([i for i in range(length)], 200)  # 2400/12
    files = os.listdir(folder_path)
    if not os.path.exists("datasets/raw/val"):
        os.makedirs("datasets/raw/val")
    for index, i in enumerate(val_index):
        os.rename(
            os.path.join(folder_path, files[i]),
            f"datasets/raw/val/val_{folder[:-6]}_{index}.JPG",
        )
    info["val"][f"{folder[:-6]}"] = len(val_index)
    info["val"]["total_count"] += len(val_index)

    # test
    length = len(os.listdir(folder_path))
    test_index = random.sample([i for i in range(length)], 200)  # 2400/12
    files = os.listdir(folder_path)
    if not os.path.exists("datasets/raw/test"):
        os.makedirs("datasets/raw/test")
    for index, i in enumerate(test_index):
        os.rename(
            os.path.join(folder_path, files[i]),
            f"datasets/raw/test/test_{folder[:-6]}_{index}.JPG",
        )
    info["val"][f"{folder[:-6]}"] = len(test_index)
    info["val"]["total_count"] += len(test_index)

print(info)
