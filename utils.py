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
    y = to_categorical(y, 12)

    XFur = X[[i for i in range(len(y)) if y[i] in []]]
    XGood = X[[i for i in range(len(y)) if y[i] in []]]
    XMis = X[[i for i in range(len(y)) if y[i] in []]]
    yFur = y[[i for i in range(len(y)) if y[i] in []]]
    yGood = y[[i for i in range(len(y)) if y[i] in []]]
    yMis = y[[i for i in range(len(y)) if y[i] in []]]

    yExp = np.array([2 if i in [] else (0 if i in [] else 1) for i in y])
    yExp = to_categorical(yExp, 3)

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
    with open(os.path.join(f"Dataset/pc", f"{type}.json")) as file:
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

    



def load_data(task,path, method, batch_size=None):
    Xtest, ytest, Xtrain, ytrain, Xval, yval = [], [], [], [], [], []

    # divide into train/validation/test dataset
    for i in ["train", "val", "test"]:
        folder_path = os.path.join(path, i)
        file = os.listdir(folder_path)
        for f in file:
            if not os.path.isfile(os.path.join(folder_path, f)):
                continue
            else:
                img = cv2.imread(os.path.join(folder_path, f))
                if "test" in f:
                    Xtest.append(img)
                    ytest.append(label_map[f"{f.split('_')[1]}"])
                elif "train" in f:
                    Xtrain.append(img)
                    ytrain.append(label_map[f"{f.split('_')[1]}"])
                elif "val" in f:
                    Xval.append(img)
                    yval.append(label_map[f"{f.split('_')[1]}"])
    
    Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
    Xval, yval = shuffle(Xval, yval, random_state=42)
    Xtest, ytest = shuffle(Xtest, ytest, random_state=42)

    # no need to shuffle, sample make it messay original
    if method in [
        "CNN",
    ]:  # customized, loaded data with batches
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
    
    elif method in ["MoE"]:
        train_dataset = MoEsplit(Xtrain, ytrain)
        val_dataset = MoEsplit(Xval, yval)
        test_dataset = MoEsplit(Xtest, ytest)
        return train_dataset,val_dataset,test_dataset
    
    elif method in ["ResNet50","InceptionV3","MobileNetV2","NASNetMobile","VGG19"]:
        Xtrain = np.array(Xtrain)
        Xval = np.array(Xval)
        Xtest = np.array(Xtest)
        ytrain = np.array(ytrain)
        yval = np.array(yval)
        ytest = np.array(ytest)
        return Xtrain, ytrain, Xtest, ytest, Xval, yval
    
    elif method == "multimodal":
        train_dataset = (Xtrain) + load_data_multimodal("train") + (ytrain)
        val_dataset = (Xval) + load_data_multimodal("val") + (yval)
        test_dataset = (Xtest) + load_data_multimodal("test") + (ytest)
        return train_dataset,val_dataset,test_dataset
    

"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(task, method, multilabel=False, lr=0.001, batch_size=32,epochs=10):
    if "CNN" in method:
        model = CNN(method, multilabel=multilabel, lr=lr)
    elif method == "MoE":
        model = MoE(method, lr=lr,batch_size=batch_size, epochs=epochs)
    elif method in ["ResNet50","InceptionV3","MobileNetV2","NASNetMobile","VGG19"]:
        model = Pretrained(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "multimodal":
        model = Multimodal(method,  lr=lr, epochs=epochs, batch_size=batch_size)

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
        disp.plot(ax=axes[index])
        disp.ax_.set_title(mode)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("")
        if index != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.45, 0.05, "Predicted label", ha="center")
    plt.subplots_adjust(wspace=0.40, hspace=0.1)
    fig.colorbar(disp.im_, ax=axes)

    if not os.path.exists("Outputs/image_classification/confusion_matrix/"):
        os.makedirs("Outputs/image_classification/confusion_matrix/")
    fig.savefig(f"Outputs/image_classification/confusion_matrix/{method}.png")
    plt.close()


"""
description: This function is used for visualizing auc roc curves.
param {*} task: task A or B
param {*} method: selected model
param {*} ytrain: train ground truth
param {*} yval: validation ground truth
param {*} ytest: test ground truth
param {*} train_pred: train prediction
param {*} val_pred: validation prediction
param {*} test_pred: test prediction
"""


def visual4auc(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    # roc curves
    rocs = {
        "train": roc_curve(
            np.array(ytrain).astype(int),
            train_pred.astype(int),
            pos_label=1,
            drop_intermediate=True,
        ),
        "val": roc_curve(
            np.array(yval).astype(int),
            val_pred.astype(int),
            pos_label=1,
            drop_intermediate=True,
        ),
        "test": roc_curve(
            np.array(ytest).astype(int),
            test_pred.astype(int),
            pos_label=1,
            drop_intermediate=True,
        ),
    }

    colors = list(mcolors.TABLEAU_COLORS.keys())

    plt.figure(figsize=(10, 6))
    for index, mode in enumerate(["train", "val", "test"]):
        plt.plot(
            rocs[mode][0],
            rocs[mode][1],
            lw=1,
            label="{}(AUC={:.3f})".format(mode, auc(rocs[mode][0], rocs[mode][1])),
            color=mcolors.TABLEAU_COLORS[colors[index]],
        )
    plt.plot([0, 1], [0, 1], "--", lw=1, color="grey")
    plt.axis("square")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title(f"ROC Curve for {method}", fontsize=10)
    plt.legend(loc="lower right", fontsize=5)

    if not os.path.exists("Outputs/image_classification/roc_curve/"):
        os.makedirs("Outputs/image_classification/roc_curve/")
    plt.savefig(f"Outputs/image_classification/roc_curve/{method}.png")
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

    if not os.path.exists("Outputs/images/"):
        os.makedirs("Outputs/images/")
    fig.savefig(f"Outputs/images/label_distribution_task{task}.png")
    plt.close()
