# -*- encoding: utf-8 -*-
"""
@File    :   data_preprocessing.py
@Time    :   2023/12/16 22:28:04
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all data preprocessing procedures for task A. 
        Notice that there are lots of comments here as trials and experiments for comparison. 
        Most of the results are explained and visualized in the report.
"""

# here put the import lib
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

# """
#     In this project, two kinds of datasets are used.
#     The first one is the npz format download from the website with provided PneumoniaMNIST and PathMNIST package.
#     The second one is the dataset saved as .png format with the commands running in terminal.
#     Detailed dataset deployment can be seen in README.md and Github link.
# """
# # to download dataset in npz format from MeMNIST website, can use the sentence below
# # dataset2 = PathMNIST(split='train',download=True,root="Datasets/")

# # how to save the dataset as png figure and csv (reference from MeMNIST)
# # python -m medmnist save --flag=pathmnist --postfix=png --folder=Datasets/ --root=Datasets/

# """
# description: This function is used for histogram equalization and comparison of CLAHE method.
# param {*} path: path of raw dataset
# param {*} f: filename
# return {*}: original image, image after histogram equalization, image after CLAHE
# """


def histogram_equalization(path, f):
    img = cv2.imread(os.path.join(path, f))
    imgResize = cv2.resize(img, (100, 100), interpolation=cv2.INTER_NEAREST)
    cl = cv2.cvtColor(imgResize, cv2.COLOR_RGB2HSV)
    # create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl[:, :, 2] = clahe.apply(cl[:, :, 2])
    cl = cv2.cvtColor(cl, cv2.COLOR_HSV2RGB)

    return imgResize, cl


"""
description: This function is used for Sobel operation.
param {*} imgEqu: image after histogram equalization
return {*}: image after Sobel operation
"""


def sobel(imgCl):
    SobelX = cv2.Sobel(imgCl, cv2.CV_16S, 1, 0)
    SobelY = cv2.Sobel(imgCl, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(SobelX)
    absY = cv2.convertScaleAbs(SobelY)
    SobelXY = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    imgSobel = np.uint8(cv2.normalize(SobelXY, None, 0, 255, cv2.NORM_MINMAX))

    passivation = (
        imgSobel * 0.3 + imgCl
    )  # sobel add on image after histogram equalization
    imgPas = np.uint8(cv2.normalize(passivation, None, 0, 255, cv2.NORM_MINMAX))
    return imgPas


# """
# description: This function is a conclusion for all data preprocessing procedures.
# param {*} raw_path: raw dataset path
# return {*}: size of train/validation/test dataset after all data preprocessing procedures.
# """


def data_preprocess():
    print("Start preprocessing data......")

    # data preprocessing
    for name in ["train", "val", "test"]:
        raw_path = f"Datasets/raw/{name}"
        os.makedirs(f"Datasets/preprocessed/{name}", exist_ok=True)
        for index, f in enumerate(os.listdir(raw_path)):
            if not os.path.isfile(os.path.join(raw_path, f)):
                continue
            else:
                imgResize, cl = histogram_equalization(raw_path, f)
                imgPas = sobel(cl)
                # need to resize all image into same size
                cv2.imwrite(
                    os.path.join(f"Datasets/preprocessed/{name}", f"{f}"), imgPas
                )
    print("Finish preprocessing data.")


"""
    Experiments for comparison of histogram equalization with single/double channel, CLAHE and RGB/HSV format.
    This part is for experiment and will not be included in the committed code
"""

# img = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")

# # comparison
# # equ without convert into HSV: color difference
# # two channel: no, color difference
# # equ: too much noise
# equ1 = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")
# equ1[:, :, 0] = cv2.equalizeHist(equ1[:, :, 0])
# equ1[:, :, 1] = cv2.equalizeHist(equ1[:, :, 1])
# equ1[:, :, 2] = cv2.equalizeHist(equ1[:, :, 2])  # no

# # CLAHE without convert into HSV
# cl1 = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl1[:, :, 0] = clahe.apply(cl1[:, :, 0])
# cl1[:, :, 1] = clahe.apply(cl1[:, :, 1])
# cl1[:, :, 2] = clahe.apply(cl1[:, :, 2])

# # equ convert into HSV and equalize two channels
# equ2 = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")
# equ2 = cv2.cvtColor(equ2, cv2.COLOR_RGB2HSV)
# equ2[:, :, 1] = cv2.equalizeHist(equ2[:, :, 1])
# equ2[:, :, 2] = cv2.equalizeHist(equ2[:, :, 2])
# equ2 = cv2.cvtColor(equ2, cv2.COLOR_HSV2RGB)  # no

# # CLAHE convert into HSV and equalize two channels
# cl2 = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")
# cl2 = cv2.cvtColor(cl2, cv2.COLOR_RGB2HSV)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl2[:, :, 1] = clahe.apply(cl2[:, :, 1])
# cl2[:, :, 2] = clahe.apply(cl2[:, :, 2])
# cl2 = cv2.cvtColor(cl2, cv2.COLOR_HSV2RGB)  # no

# # equ convert into HSV and equalize one channel
# equ3 = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")
# equ3 = cv2.cvtColor(equ3, cv2.COLOR_RGB2HSV)
# equ3[:, :, 2] = cv2.equalizeHist(equ3[:, :, 2])
# equ3 = cv2.cvtColor(equ3, cv2.COLOR_HSV2RGB)  # no

# # CLAHE convert into HSV and equalize one channel (final select)
# cl3 = cv2.imread("Datasets/raw/train/train_bicycle_11.JPG")
# cl3 = cv2.cvtColor(cl3, cv2.COLOR_RGB2HSV)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl3[:, :, 2] = clahe.apply(cl3[:, :, 2])
# cl3 = cv2.cvtColor(cl3, cv2.COLOR_HSV2RGB)

# res = np.hstack((img, equ1, cl1, equ2, cl2, equ3, cl3))  # stacking images side-by-side
# cv2.imwrite(os.path.join("Outputs/", "res.png"), res)
# cv2.imshow("result", res)
# cv2.waitKey(0)


# chans = cv2.split(img)
# colors = ("b", "g", "r")
# plt.subplot(1, 2, 1)
# plt.title("calcHist before equalization")
# plt.xlabel("Bins")
# plt.ylabel("Pixels Num")

# for chan, color in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     print(color)
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])

# # the selected one
# chans = cv2.split(cl3)
# colors = ("b", "g", "r")
# plt.subplot(1, 2, 2)
# plt.title("calcHist after equalization")
# plt.xlabel("Bins")
# plt.ylabel("Pixels Num")

# for chan, color in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     print(color)
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])

# plt.show()
# plt.savefig("Outputs/histogram_example.png")

"""
    Experiments for comparison of CLAHE, Laplacian, Sobel operation, Gamma correction, etc.
    This part is for experiment and will not be included in the committed code
"""
# experiment 2: selection for sobel/lapacian/gamma correction
# result：sobel
# laplacian may lead to lots of noise
# gamma can improve light but make the image color become shallow

# img = cv2.imread("Datasets/raw/train/train_cabinet_215.JPG")
# cl = cv2.imread("Datasets/raw/train/train_cabinet_215.JPG")
# cl = cv2.cvtColor(cl, cv2.COLOR_RGB2HSV)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# cl[:, :, 2] = clahe.apply(cl[:, :, 2])
# cl = cv2.cvtColor(cl, cv2.COLOR_HSV2RGB)
# # laplacian

# # laplacian
# kernel_cl = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.int8)
# laplacian_cl = cv2.filter2D(cl, ddepth=-1, kernel=kernel_cl)
# imglaplacian_cl = np.uint8(cv2.normalize(laplacian_cl, None, 0, 255, cv2.NORM_MINMAX))
# addlap_cl = cl + imglaplacian_cl
# addlap_cl = np.uint8(cv2.normalize(addlap_cl, None, 0, 255, cv2.NORM_MINMAX))

# # sobel
# SobelX_cl = cv2.Sobel(cl, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
# SobelY_cl = cv2.Sobel(cl, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
# absX_cl = cv2.convertScaleAbs(SobelX_cl)  # 转回 uint8
# absY_cl = cv2.convertScaleAbs(SobelY_cl)  # 转回 uint8
# SobelXY_cl = cv2.addWeighted(absX_cl, 0.5, absY_cl, 0.5, 0)  # 用绝对值近似平方根
# imgSobel_cl = np.uint8(cv2.normalize(SobelXY_cl, None, 0, 255, cv2.NORM_MINMAX))

# passivation1_cl = (
#     imgSobel_cl * 0.3 + cl
# )  # sobel add on image after histogram equalization
# passivation2_cl = imgSobel_cl * 0.3 + addlap_cl  # sobel add on image after laplacian
# imgPassi1_cl = np.uint8(cv2.normalize(passivation1_cl, None, 0, 255, cv2.NORM_MINMAX))
# imgPassi2_cl = np.uint8(cv2.normalize(passivation2_cl, None, 0, 255, cv2.NORM_MINMAX))

# # gamma
# epsilon = 1e-5
# Gamma1_cl = np.power(imgPassi1_cl + epsilon, 0.5)
# Gamma2_cl = np.power(imgPassi2_cl + epsilon, 0.5)
# imgGamma1_cl = np.uint8(cv2.normalize(Gamma1_cl, None, 0, 255, cv2.NORM_MINMAX))
# imgGamma2_cl = np.uint8(cv2.normalize(Gamma2_cl, None, 0, 255, cv2.NORM_MINMAX))

# res = np.hstack(
#     (img, cl, addlap_cl, imgPassi1_cl, imgPassi2_cl, imgGamma1_cl, imgGamma2_cl)
# )  # stacking images side-by-side
# cv2.imwrite(os.path.join("Outputs/", "res.png"), res)
# cv2.imshow("result", res)
# cv2.waitKey(0)

# lapalican add noise cannot use


# data_preprocess()
