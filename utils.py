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


"""
description: This function is used for loading data from preprocessed dataset into model input.
param {*} task: task Aor B
param {*} path: preprocessed dataset path
param {*} method: selected model for experiment
param {*} batch_size: batch size of NNs
return {*}: loaded model input 
"""


def load_data(task, path, method, batch_size=None):
    Xtest, ytest, Xtrain, ytrain, Xval, yval = [], [], [], [], [], []
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
    # Xtrain = Xtrain[:10]
    # ytrain = ytrain[:10]

    # if method in ["LR", "KNN", "SVM", "DT", "NB", "RF", "ABC", "KMeans"]:  # baselines
    #     if task == "A":
    #         n, h, w = np.array(Xtrain).shape
    #         Xtrain = np.array(Xtrain).reshape(
    #             n, h * w
    #         )  # need to reshape gray picture into two-dimensional ones
    #         Xval = np.array(Xval).reshape(len(Xval), h * w)
    #         Xtest = np.array(Xtest).reshape(len(Xtest), h * w)
    #     elif task == "B":
    #         n, h, w, c = np.array(Xtrain).shape
    #         Xtrain = np.array(Xtrain).reshape(n, h * w * c)
    #         Xval = np.array(Xval).reshape(len(Xval), h * w * c)
    #         Xtest = np.array(Xtest).reshape(len(Xtest), h * w * c)

    #         # shuffle dataset
    #         Xtrain, ytrain = shuffle(Xtrain, ytrain, random_state=42)
    #         Xval, yval = shuffle(Xval, yval, random_state=42)
    #         Xtest, ytest = shuffle(Xtest, ytest, random_state=42)

    #         # use PCA for task B to reduce dimensionality
    #         pca = PCA(n_components=64)
    #         Xtrain = pca.fit_transform(Xtrain)
    #         Xval = pca.fit_transform(Xval)
    #         Xtest = pca.fit_transform(Xtest)

    #     return Xtrain, ytrain, Xtest, ytest, Xval, yval

    # else:  # pretrained or customized
    #     n, h, w, c = np.array(Xtrain).shape
    #     Xtrain = np.array(Xtrain)
    #     Xval = np.array(Xval)
    #     Xtest = np.array(Xtest)

    #     """
    #         Notice that due to large size of task B dataset, part of train and validation data is sampled for
    #         pretrained network. However, all test data are used for performance measurement in testing procedure.
    #     """
    #     if task == "B":
    #         sample_index = random.sample([i for i in range(Xtrain.shape[0])], 40000)
    #         Xtrain = Xtrain[sample_index, :, :, :]
    #         ytrain = np.array(ytrain)[sample_index].tolist()

    #         sample_index_val = random.sample([i for i in range(Xval.shape[0])], 5000)
    #         Xval = Xval[sample_index_val, :]
    #         yval = np.array(yval)[sample_index_val].tolist()

    #         sample_index_test = random.sample([i for i in range(Xtest.shape[0])], 7180)
    #         Xtest = Xtest[sample_index_test, :]
    #         ytest = np.array(ytest)[sample_index_test].tolist()

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
    else:
        return Xtrain, ytrain, Xtest, ytest, Xval, yval


"""
description: This function is used for loading selected model.
param {*} task: task A or B
param {*} method: selected model
param {*} multilabel: whether configuring multilabels setting (can only be used with MLP/CNN in task B)
param {*} lr: learning rate for adjustment and tuning
return {*}: constructed model
"""


def load_model(task, method, multilabel=False, lr=0.001):
    if "CNN" in method:
        model = CNN(method, multilabel=multilabel, lr=lr)
    # elif "DenseNet201" in method:
    #     model = A_DenseNet201(method) if task == "A" else B_DenseNet201(method)
    # elif "InceptionV3" in method:
    #     model = A_InceptionV3(method) if task == "A" else B_InceptionV3(method)
    # elif "MLP" in method:
    #     model = (
    #         A_MLP(task, method, lr=lr)
    #         if task == "A"
    #         else B_MLP(task, method, multilabel=multilabel, lr=lr)
    #     )
    # elif "MobileNetV2" in method:
    #     model = A_MobileNetV2(method) if task == "A" else B_MobileNetV2(method)
    # elif "ResNet50" in method:
    #     model = A_ResNet50(method) if task == "A" else B_ResNet50(method)
    # elif "VGG16" in method:
    #     model = A_VGG16(method) if task == "A" else B_VGG16(method)
    # elif method == "EnsembleNet":
    #     model = A_EnsembleNet(lr=lr) if task == "A" else B_EnsembleNet(lr=lr)
    # elif method == "KMeans":
    #     model = A_KMeans() if task == "A" else B_KMeans()
    # else:  # baselines
    #     model = A_Baselines(method) if task == "A" else B_Baselines(method)

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
