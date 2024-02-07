# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2023/12/16 22:44:21
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for all utils function like visualization, data loading, model loading, etc.
"""

# here put the import lib
import os
import cv2
import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.colors as mcolors
from image_classification.ViT import ViT
from image_generation.AutoEncoder import AutoEncoder
from patchify import patchify
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    silhouette_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
    auc,
)
from image_classification.CNN import CNN
from image_classification.MoE import MoE
from tensorflow.keras.utils import to_categorical
from image_classification.Multimodal import Multimodal

from image_classification.pretrained import Pretrained
from image_generation.BaseGAN import BaseGAN
from image_generation.ConGAN import ConGAN
from image_generation.PencilGAN import PencilGAN
from image_classification.AdvCNN import AdvCNN


"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""

label_map = {
        "bicycle": 0,
        "cabinet": 1,
        "chair": 2,
        "coffee": 3,
        "fan": 4,
        "kettle": 5,
        "lamp": 6,
        "mug": 7,
        "sofa": 8,
        "stapler": 9,
        "table": 10,
        "toaster": 11,
    }

def MoEsplit(X,y):
    X = np.array(X).astype('float32')/255   

    XFur = X[[i for i in range(len(y)) if y[i] in [1,6,10]]]
    XGood = X[[i for i in range(len(y)) if y[i] in [0,3,4,5,7,9,11]]]
    XMis = X[[i for i in range(len(y)) if y[i] in [2,8]]]
    yFur = np.array(y)[[i for i in range(len(y)) if y[i] in [1,6,10]]]
    yGood = np.array(y)[[i for i in range(len(y)) if y[i] in [0,3,4,5,7,9,11]]]
    yMis = np.array(y)[[i for i in range(len(y)) if y[i] in [2,8]]]
    yFur = to_categorical(yFur, 12)
    yGood = to_categorical(yGood, 12)
    yMis = to_categorical(yMis, 12)

    yExp = np.array([2 if i in [2,8] else (0 if i in [1,6,10] else 1) for i in y])
    y = np.array(y)
    y = to_categorical(y, 12)
    yExp = to_categorical(yExp, 3)
    print(type(XFur))
    print(type(X))

    return X, XFur, XGood, XMis, \
            y, yExp, yFur, yGood, yMis

def load_data_multimodal(type=None):
    # seg
    X_seg,  X_contour, X_pc, y_seg, y_contour, y_pc = [],[],[],[],[],[]
    for i in ["segmented","contour"]:
        folder_path = os.path.join(f"Dataset/{i}", type)
        file = os.listdir(folder_path)
        for f in file:
            if not os.path.isfile(os.path.join(folder_path, f)):
                continue
            else:
                img = cv2.imread(os.path.join(folder_path, f))
                if i == "segmented":
                    X_seg.append(img)
                    y_seg.append(label_map[f"{f.split('_')[1]}"])
                elif i == "contour":
                    X_contour.append(img)
                    y_contour.append(label_map[f"{f.split('_')[1]}"])
               
    # pc
    with open(os.path.join(f"datasets/pc", f"{type}.json")) as file:
        features = json.load(file)
        sorted_keys = sorted(features.keys())
        for key in sorted_keys:
            coor = features[key]["coordinates"] # 4096*3
            R = features[key]["channels"]["R"]
            G = features[key]["channels"]["G"]
            B = features[key]["channels"]["B"]
            channel = np.concatenate((R,G,B),axis=1)  # 4096*3
            feature = np.concatenate((coor,channel),axis=1)  # 4096*6
            X_pc.append(feature)
            y_pc.append(label_map[f"{key.split('_')[1]}"])
    print(y_seg)
    print(y_contour)
    print(y_pc)
    
    return  X_seg,  X_contour, X_pc


def load_data_augmented(type="train"):
    X,y = [], []
    folder_path = os.path.join("datasets/augmented/", type)
    file = os.listdir(folder_path)
    for f in file:
        if not os.path.isfile(os.path.join(folder_path, f)):
            continue
        else:
            img = cv2.imread(os.path.join(folder_path, f))
            X.append(img)
            y.append(label_map[f"{f.split('_')[1]}"])

    return X,y

def load_batch(Xtrain,ytrain,Xval,yval,Xtest,ytest,batch_size = None):
    train_ds = tf.data.Dataset.from_tensor_slices(
        (Xtrain, np.array(ytrain).astype(int))
    ).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (Xval, np.array(yval).astype(int))
    ).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (Xtest, np.array(ytest).astype(int))
    ).batch(batch_size)
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)  # normalization
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds

def load_patches(X):
    X_patches = patchify(X[0], (10,10,3), 10)
    X_save = np.reshape(X_patches, (100,10,10,3))
    if not os.path.exists("outputs/image_classification/ViT/"):
        os.makedirs("outputs/image_classification/ViT/")
    for index,patch in enumerate(X_save):
        # print(patch)
        cv2.imwrite(f"outputs/image_classification/ViT/patch_{index}.JPG",patch)
    # print(X_patches)
    X_patches = np.reshape(X_patches, (1,100,10*10*3))
    for x in X[1:]:
        patches = patchify(x, (10,10,3), 10)
        patches = np.reshape(patches, (1,100,10*10*3))
        X_patches = np.concatenate(((np.array(X_patches)), patches), axis=0)
    return X_patches.astype(np.int64)

def sample_ViT(X,y,n):
    ViT_index,ViT_label = [],[]
    for i in range(12):
        class_index = [index for index, j in enumerate(y) if label_map[n[index]] == i]
        ViT_index += random.sample(class_index, 100)
    ViT_sample = [i for index,i in enumerate(X) if index in ViT_index]
    ViT_label = [i for index,i in enumerate(y) if index in ViT_index]
    return ViT_sample,ViT_label

def load_data(task,path, method, batch_size=None):
    Xtest, ytest, Xtrain, ytrain, Xval, yval, ntrain, nval, ntest = [], [], [], [], [], [],[],[],[]

    # divide into train/validation/test dataset
    for i in ["train", "val", "test"]:
        folder_path = os.path.join(path, i)
        file = os.listdir(folder_path)
        for f in file:
            if not os.path.isfile(os.path.join(folder_path, f)):
                continue
            else:
                img = cv2.imread(os.path.join(folder_path, f))
                if method in ["BaseGAN","ConGAN","PencilGAN","AutoEncoder"]:
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                if "test" in f:
                    Xtest.append(img)
                    ytest.append(label_map[f"{f.split('_')[1]}"])
                    ntest.append(f.split('_')[1])
                elif "train" in f:
                    Xtrain.append(img)
                    ytrain.append(label_map[f"{f.split('_')[1]}"])
                    ntrain.append(f.split('_')[1])
                elif "val" in f:
                    Xval.append(img)
                    yval.append(label_map[f"{f.split('_')[1]}"])
                    nval.append(f.split('_')[1])
    if method == "ViT":  # 1200,1200,1200
        Xtrain,ytrain = sample_ViT(Xtrain,ytrain,ntrain)
        Xval,yval = sample_ViT(Xval,yval,nval)
        Xtest,ytest = sample_ViT(Xtest,ytest,ntest)
    
        # Xtrain_index = random.sample([i for i in range(len(Xtrain))],)
        # Xtrain,ytrain = Xtrain[:1000],ytrain[:1000]
        # Xval,yval = Xval[:1000],yval[:1000]
        # Xtest,ytest = Xtest[:1000],ytest[:1000]
    
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
    Xval, yval = shuffle(Xval, yval, random_state=42)
    Xtest, ytest = shuffle(Xtest, ytest, random_state=42)

    # no need to shuffle, sample make it messay original
    # if method in ["CNN","ViT"]:
    #     # if method == "ViT":
    #     #     Xtrain_patches = load_patches(Xtrain)
    #     #     Xval_patches = load_patches(Xval)
    #     #     Xtest_patches = load_patches(Xtest)
    #     #     train_ds, val_ds, test_ds = load_batch(Xtrain_patches,ytrain,Xval_patches,yval,\
    #     #                                             Xtest_patches,ytest,batch_size = batch_size)
        
    #     else:
    #         train_ds, val_ds, test_ds = load_batch(Xtrain,ytrain,Xval,yval,\
    #                                                 Xtest,ytest,batch_size = batch_size)
    #     return train_ds, val_ds, test_ds
    if method in ["CNN"]:
       
        train_ds, val_ds, test_ds = load_batch(Xtrain,ytrain,Xval,yval,\
                                                    Xtest,ytest,batch_size = batch_size)
        return train_ds, val_ds, test_ds
    
    elif method in ["MoE"]:
        train_dataset = MoEsplit(Xtrain, ytrain)
        val_dataset = MoEsplit(Xval, yval)
        test_dataset = MoEsplit(Xtest, ytest)
        return train_dataset,val_dataset,test_dataset
    
    elif method in ["ViT","ConGAN","AutoEncoder","PencilGAN","BaseGAN","ResNet50","InceptionV3","MobileNetV2","NASNetMobile","VGG19"]:
        if method == "ViT":
            Xtrain = load_patches(Xtrain)
            Xval = load_patches(Xval)
            Xtest = load_patches(Xtest)
        Xtrain = np.array(Xtrain)
        Xval = np.array(Xval)
        Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        yval = np.array(yval)
        ytest = np.array(ytest)

        if method in ["AutoEncoder","ConGAN"]:
            Xtrain = Xtrain/255
            Xval = Xval/255
            Xtest = Xtest/255
        return Xtrain, ytrain, Xtest, ytest, Xval, yval
    
    elif method == "Multimodal":
        train_dataset = (Xtrain) + load_data_multimodal("train") + (ytrain)
        val_dataset = (Xval) + load_data_multimodal("val") + (yval)
        test_dataset = (Xtest) + load_data_multimodal("test") + (ytest)
        return train_dataset,val_dataset,test_dataset
    elif method == "AdvCNN":
        Xtrain = np.array(Xtrain + load_data_augmented("train")[0])
        Xval = np.array(Xval + load_data_augmented("val")[0])
        Xtest = np.array(Xtest + load_data_augmented("test")[0])
        ytrain = np.array(ytrain + load_data_augmented("train")[1])
        yval = np.array(yval + load_data_augmented("val")[1])
        ytest = np.array(ytest + load_data_augmented("test")[1])
        return Xtrain, ytrain, Xtest, ytest, Xval, yval




    

"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(task, method, multilabel=False, lr=0.001, batch_size=32,epochs=10):
    if method == "CNN":
        model = CNN(method, multilabel=multilabel, lr=lr)
    elif method == "MoE":
        model = MoE(method, lr=lr,batch_size=batch_size, epochs=epochs)
    elif method in ["ResNet50","InceptionV3","MobileNetV2","NASNetMobile","VGG19"]:
        model = Pretrained(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "Multimodal":
        model = Multimodal(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "BaseGAN":
        model = BaseGAN(method,lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "PencilGAN":
        model = PencilGAN(method,lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "ConGAN":
        model = ConGAN(method,lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "AdvCNN":
        model = AdvCNN(method,lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "ViT":
        model = ViT(method,lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "AutoEncoder":
        model = AutoEncoder(method,lr=lr, epochs=epochs, batch_size=batch_size)

    return model


"""
description: This function is used for visualizing confusion matrix.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


def visual4cm(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    # confusion matrix
    cms = {
        "train": confusion_matrix(ytrain, train_pred),
        "val": confusion_matrix(yval, val_pred),
        "test": confusion_matrix(ytest, test_pred),
    }

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey="row")
    for index, mode in enumerate(["train", "val", "test"]):
        disp = ConfusionMatrixDisplay(
            cms[mode], display_labels=sorted(list(set(ytrain)))
        )
        # print(sorted(list(set(ytrain))))
        # print(cms[mode])
        disp.plot(ax=axes[index])
        # disp.plot()
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists("outputs/image_classification/confusion_matrix/"):
        os.makedirs("outputs/image_classification/confusion_matrix/")
    fig.savefig(f"outputs/image_classification/confusion_matrix/{method}.png")
    plt.close()


"""
description: This function is used for calculating metrics performance including accuracy, precision, recall, f1-score.
param {*} task: task A or B
param {*} y: ground truth
param {*} pred: predicted labels
"""


def get_metrics(task, y, pred):
    result = {
        "acc": round(
            accuracy_score(np.array(y).astype(int), pred.astype(int)) * 100, 4
        ),
        "pre": round(
            precision_score(np.array(y).astype(int), pred.astype(int), average="macro")
            * 100,
            4,
        ),
        "rec": round(
            recall_score(np.array(y).astype(int), pred.astype(int), average="macro")
            * 100,
            4,
        ),
        "f1": round(
            f1_score(np.array(y).astype(int), pred.astype(int), average="macro") * 100,
            4,
        ),
    }
    return result


"""
description: This function is used for visualizing dataset label distribution.
param {*} task: task A or B
param {*} data: npz data
"""


def visual4label(task, data):
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(6, 3), subplot_kw=dict(aspect="equal"), dpi=600
    )

    for index, mode in enumerate(["train", "val", "test"]):
        pie_data = [
            np.count_nonzero(data[f"{mode}_labels"].flatten() == i)
            for i in range(len(set(data[f"{mode}_labels"].flatten().tolist())))
        ]
        labels = [
            f"label {i}"
            for i in sorted(list(set(data[f"{mode}_labels"].flatten().tolist())))
        ]
        wedges, texts, autotexts = ax[index].pie(
            pie_data,
            autopct=lambda pct: f"{pct:.2f}%\n({int(np.round(pct/100.*np.sum(pie_data))):d})",
            textprops=dict(color="w"),
        )
        if index == 2:
            ax[index].legend(
                wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
            )
        size = 6 if task == "A" else 3
        plt.setp(autotexts, size=size, weight="bold")
        ax[index].set_title(mode)
    plt.tight_layout()

    if not os.path.exists("outputs/images/"):
        os.makedirs("outputs/images/")
    fig.savefig(f"outputs/images/label_distribution_task{task}.png")
    plt.close()


def visual4loss(method,type,loss, acc):
    plt.figure()
    plt.title(f"Loss for epochs of {method}")
    plt.plot(range(len(loss)),loss,color='pink', linestyle='dashed', \
             marker='o', markerfacecolor='grey',markersize=10)
    plt.tight_layout()

    if not os.path.exists("outputs/image_classification/metric_lines"):
        os.makedirs("outputs/image_classification/metric_lines")
    plt.savefig(f"outputs/image_classification/metric_lines/{method}_{type}_loss.png")
    plt.close()

    plt.figure()
    plt.title(f"Accuracy for epochs of {method}")
    plt.plot(range(len(acc)),loss,color='pink', linestyle='dashed', \
             marker='o', markerfacecolor='grey',markersize=10)
    plt.tight_layout()
    plt.savefig(f"outputs/image_classification/metric_lines/{method}_{type}_acc.png")
    plt.close()

    