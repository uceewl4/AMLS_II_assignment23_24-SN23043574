# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/02/24 20:47:39
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This is the file used for all utility tools like loading data, model, visualizations.
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
from tensorflow.keras.utils import to_categorical
from patchify import patchify
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from image_classification.CNN import CNN
from image_classification.MoE import MoE
from image_classification.Multimodal import Multimodal
from image_classification.pretrained import Pretrained
from image_generation.BaseGAN import BaseGAN
from image_generation.ConGAN import ConGAN
from image_generation.PencilGAN import PencilGAN
from image_classification.AdvCNN import AdvCNN

# label map for 12 classes
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


def MoEsplit(X, y):
    """
    description: This method is used for spliting original dataset into corresonding multi-agent datasets.
    param {*} X: data
    param {*} y: label
    return {*}: original, furniture, home goods, misclassified data, original label, triple expert label,
    furniture, home good and misclassified labels
    """
    X = np.array(X).astype("float32") / 255  # normalization
    XFur = X[[i for i in range(len(y)) if y[i] in [1, 6, 10]]]
    XGood = X[[i for i in range(len(y)) if y[i] in [0, 3, 4, 5, 7, 9, 11]]]
    XMis = X[[i for i in range(len(y)) if y[i] in [2, 8]]]

    yFur = np.array(y)[[i for i in range(len(y)) if y[i] in [1, 6, 10]]]
    yGood = np.array(y)[[i for i in range(len(y)) if y[i] in [0, 3, 4, 5, 7, 9, 11]]]
    yMis = np.array(y)[[i for i in range(len(y)) if y[i] in [2, 8]]]
    yFur = to_categorical(yFur, 12)
    yGood = to_categorical(yGood, 12)
    yMis = to_categorical(yMis, 12)

    yExp = np.array([2 if i in [2, 8] else (0 if i in [1, 6, 10] else 1) for i in y])
    y = np.array(y)
    y = to_categorical(y, 12)
    yExp = to_categorical(yExp, 3)

    return X, XFur, XGood, XMis, y, yExp, yFur, yGood, yMis


def load_data_multimodal(type=None):
    """
    description: This method is used for loading multimodal data.
    param {*} type: train, validation or test
    return {*}: loaded data for segmentation, contour, point cloud
    """
    # segmentation/contour: (6000,100,100,3)
    X_seg, X_contour, X_pc, y_seg, y_contour, y_pc = [], [], [], [], [], []
    for i in ["segmented", "contour"]:
        folder_path = os.path.join(f"datasets/{i}", type)
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

    # point cloud
    with open(os.path.join(f"datasets/pc", f"{type}.json")) as file:
        features = json.load(file)
        for i in os.listdir(folder_path):
            coor = features[i]["coordinates"]  # 4096, 3
            R = features[i]["channels"]["R"]
            G = features[i]["channels"]["G"]
            B = features[i]["channels"]["B"]
            channel = np.concatenate(
                (
                    np.array(R).reshape(-1, 1),
                    np.array(G).reshape(-1, 1),
                    np.array(B).reshape(-1, 1),
                ),
                axis=1,
            )  # 4096, 3
            feature = np.concatenate((coor, channel), axis=1).reshape(
                64, 64, 6
            )  # 4096, 6
            X_pc.append(feature)
            y_pc.append(label_map[f"{i.split('_')[1]}"])

    return X_seg, X_contour, X_pc


def load_data_augmented(type="train"):
    """
    description: This method is used for loading augmentated data for adversarial attacks.
    param {*} type: train, validation or test
    return {*}: loaded data
    """
    X, y = [], []
    folder_path = os.path.join("datasets/augmented/", type)
    file = os.listdir(folder_path)
    for f in file:
        if not os.path.isfile(os.path.join(folder_path, f)):
            continue
        else:
            img = cv2.imread(os.path.join(folder_path, f))
            X.append(img)
            y.append(label_map[f"{f.split('_')[1]}"])
    return X, y


def load_batch(Xtrain, ytrain, Xval, yval, Xtest, ytest, batch_size=None):
    """
    description: This method is used for loading data into batches.
    param {*} Xtrain: train data
    param {*} ytrain: train label
    param {*} Xval: validation data
    param {*} yval: validation label
    param {*} Xtest: test data
    param {*} ytest: test label
    param {*} batch_size
    return {*}: dataloaders
    """
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
    """
    description: This method is used for ViT to get image patches.
    param {*} X: input images
    return {*}: patches
    """
    X_patches = patchify(X[0], (10, 10, 3), 10)  # 100 for 10x10x3
    X_save = np.reshape(X_patches, (100, 10, 10, 3))
    if not os.path.exists("outputs/image_classification/ViT/"):
        os.makedirs("outputs/image_classification/ViT/")
    for index, patch in enumerate(X_save):
        cv2.imwrite(f"outputs/image_classification/ViT/patch_{index}.JPG", patch)

    X_patches = np.reshape(X_patches, (1, 100, 10 * 10 * 3))
    for x in X[1:]:
        patches = patchify(x, (10, 10, 3), 10)
        patches = np.reshape(patches, (1, 100, 10 * 10 * 3))
        X_patches = np.concatenate(((np.array(X_patches)), patches), axis=0)

    return X_patches.astype(np.int64)


def sample_ViT(X, y, n):
    """
    description: Due to large size of patches, this method is used for sampling from patches to
    reduce dimensionality.
    param {*} X: input data
    param {*} y: input label
    param {*} n: class name
    return {*}: sampled data and label
    """
    ViT_index, ViT_label = [], []
    for i in range(12):
        class_index = [index for index, j in enumerate(y) if label_map[n[index]] == i]
        ViT_index += random.sample(class_index, 100)
    ViT_sample = [i for index, i in enumerate(X) if index in ViT_index]
    ViT_label = [i for index, i in enumerate(y) if index in ViT_index]
    return ViT_sample, ViT_label


def load_data(task, path, method, batch_size=None):
    """
    description: This method is used for loading data for different tasks.
    param {*} task: IC or IG
    param {*} path: path of preprocessed data
    param {*} method: selected model
    param {*} batch_size
    return {*}: loaded data
    """
    Xtest, ytest, Xtrain, ytrain, Xval, yval, ntrain, nval, ntest = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # divide into train/validation/test dataset
    for i in ["train", "val", "test"]:
        folder_path = os.path.join(path, i)
        file = os.listdir(folder_path)
        for f in file:
            if not os.path.isfile(os.path.join(folder_path, f)):
                continue
            else:
                img = cv2.imread(os.path.join(folder_path, f))
                if method in ["BaseGAN", "ConGAN", "PencilGAN", "AutoEncoder"]:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if "test" in f:
                    Xtest.append(img)
                    ytest.append(label_map[f"{f.split('_')[1]}"])
                    ntest.append(f.split("_")[1])
                elif "train" in f:
                    Xtrain.append(img)
                    ytrain.append(label_map[f"{f.split('_')[1]}"])
                    ntrain.append(f.split("_")[1])
                elif "val" in f:
                    Xval.append(img)
                    yval.append(label_map[f"{f.split('_')[1]}"])
                    nval.append(f.split("_")[1])

    if method == "ViT":  # 1200,1200,1200
        Xtrain, ytrain = sample_ViT(Xtrain, ytrain, ntrain)
        Xval, yval = sample_ViT(Xval, yval, nval)
        Xtest, ytest = sample_ViT(Xtest, ytest, ntest)

    if method != "Multimodal":
        Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
        Xval, yval = shuffle(Xval, yval, random_state=42)
        Xtest, ytest = shuffle(Xtest, ytest, random_state=42)

    if method in ["CNN"]:

        train_ds, val_ds, test_ds = load_batch(
            Xtrain, ytrain, Xval, yval, Xtest, ytest, batch_size=batch_size
        )
        return train_ds, val_ds, test_ds

    elif method in ["MoE"]:
        train_dataset = MoEsplit(Xtrain, ytrain)
        val_dataset = MoEsplit(Xval, yval)
        test_dataset = MoEsplit(Xtest, ytest)
        return train_dataset, val_dataset, test_dataset

    elif method in [
        "ViT",
        "ConGAN",
        "AutoEncoder",
        "PencilGAN",
        "BaseGAN",
        "ResNet50",
        "InceptionV3",
        "MobileNetV2",
        "NASNetMobile",
        "VGG19",
    ]:
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

        if method in [
            "AutoEncoder",
            "ConGAN",
            "BaseGAN",
            "PencilGAN",
        ]:  # standardization
            Xtrain = Xtrain / 255
            Xval = Xval / 255
            Xtest = Xtest / 255
        return Xtrain, ytrain, Xtest, ytest, Xval, yval

    elif method == "Multimodal":
        Xtrain_seg, Xtrain_contour, Xtrain_pc = load_data_multimodal("train")
        Xval_seg, Xval_contour, Xval_pc = load_data_multimodal("val")
        Xtest_seg, Xtest_contour, Xtest_pc = load_data_multimodal("test")
        # shuffle
        Xtrain, Xtrain_seg, Xtrain_contour, Xtrain_pc, ytrain = shuffle(
            Xtrain, Xtrain_seg, Xtrain_contour, Xtrain_pc, ytrain, random_state=42
        )
        Xval, Xval_seg, Xval_contour, Xval_pc, yval = shuffle(
            Xval, Xval_seg, Xval_contour, Xval_pc, yval, random_state=42
        )
        Xtest, Xtest_seg, Xtest_contour, Xtest_pc, ytest = shuffle(
            Xtest, Xtest_seg, Xtest_contour, Xtest_pc, ytest, random_state=42
        )
        train_dataset = [
            np.array(Xtrain),
            np.array(Xtrain_seg),
            np.array(Xtrain_contour),
            np.array(Xtrain_pc),
            np.array(ytrain),
        ]
        val_dataset = [
            np.array(Xval),
            np.array(Xval_seg),
            np.array(Xval_contour),
            np.array(Xval_pc),
            np.array(yval),
        ]
        test_dataset = [
            np.array(Xtest),
            np.array(Xtest_seg),
            np.array(Xtest_contour),
            np.array(Xtest_pc),
            np.array(ytest),
        ]
        return train_dataset, val_dataset, test_dataset

    elif method == "AdvCNN":
        Xtrain = np.array(Xtrain + load_data_augmented("train")[0])
        Xval = np.array(Xval + load_data_augmented("val")[0])
        Xtest = np.array(Xtest + load_data_augmented("test")[0])
        ytrain = np.array(ytrain + load_data_augmented("train")[1])
        yval = np.array(yval + load_data_augmented("val")[1])
        ytest = np.array(ytest + load_data_augmented("test")[1])
        return Xtrain, ytrain, Xtest, ytest, Xval, yval


def load_model(task, method, multilabel=False, lr=0.001, batch_size=32, epochs=10):
    """
    description: This function is used for loading selected model.
    param {*} task: IC/IG
    param {*} method: selected model
    param {*} multilabel: whether configuring multilabels setting (only used for CNN)
    param {*} lr: learning rate for adjustment and tuning
    param {*} batch_size: batch size for adjustment and tuning
    param {*} lr: epochs for adjustment and tuning
    return {*}: constructed model
    """
    if method == "CNN":
        model = CNN(method, multilabel=multilabel, lr=lr)
    elif method == "MoE":
        model = MoE(method, lr=lr, batch_size=batch_size, epochs=epochs)
    elif method in ["ResNet50", "InceptionV3", "MobileNetV2", "NASNetMobile", "VGG19"]:
        model = Pretrained(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "Multimodal":
        model = Multimodal(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "BaseGAN":
        model = BaseGAN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "PencilGAN":
        model = PencilGAN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "ConGAN":
        model = ConGAN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "AdvCNN":
        model = AdvCNN(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "ViT":
        model = ViT(method, lr=lr, epochs=epochs, batch_size=batch_size)
    elif method == "AutoEncoder":
        model = AutoEncoder(method, lr=lr, epochs=epochs, batch_size=batch_size)

    return model


def visual4cm(task, method, ytrain, yval, ytest, train_pred, val_pred, test_pred):
    """
    description: This function is used for visualizing confusion matrix.
    param {*} task: IC/IG
    param {*} method: selected model
    param {*} ytrain: train ground truth
    param {*} yval: validation ground truth
    param {*} ytest: test ground truth
    param {*} train_pred: train prediction
    param {*} val_pred: validation prediction
    param {*} test_pred: test prediction
    """
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

    # save
    if not os.path.exists("outputs/image_classification/confusion_matrix/"):
        os.makedirs("outputs/image_classification/confusion_matrix/")
    fig.savefig(f"outputs/image_classification/confusion_matrix/{method}.png")
    plt.close()


def get_metrics(task, y, pred):
    """
    description: This function is used for calculating metrics performance including accuracy, precision, recall, f1-score.
    param {*} task: IC/IG
    param {*} y: ground truth
    param {*} pred: predicted labels
    """
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


def visual4loss(method, type, loss, acc):
    """
    description: This method is used for visualizing loss and accuracy along epochs.
    param {*} method: selected model
    param {*} type: train/validation/test
    param {*} loss
    param {*} acc
    """
    # loss
    plt.figure()
    plt.title(f"Loss for epochs of {method}")
    plt.plot(
        range(len(loss)),
        loss,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.tight_layout()
    if not os.path.exists("outputs/image_classification/metric_lines"):
        os.makedirs("outputs/image_classification/metric_lines")
    plt.savefig(f"outputs/image_classification/metric_lines/{method}_{type}_loss.png")
    plt.close()

    # accuracy
    plt.figure()
    plt.title(f"Accuracy for epochs of {method}")
    plt.plot(
        range(len(acc)),
        acc,
        color="pink",
        linestyle="dashed",
        marker="o",
        markerfacecolor="grey",
        markersize=10,
    )
    plt.tight_layout()
    plt.savefig(f"outputs/image_classification/metric_lines/{method}_{type}_acc.png")
    plt.close()
