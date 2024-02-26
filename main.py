# -*- encoding: utf-8 -*-
"""
@File    :   main.py
@Time    :   2024/02/24 20:40:36
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This is the main file for launching the system.
"""

# here put the import lib

import os
import argparse
import warnings
import tensorflow as tf
from utils import get_metrics, load_data, load_model, visual4cm, visual4loss

warnings.filterwarnings("ignore")

"""
    This is the part for CPU and GPU setting. Notice that part of the project 
    code is run on UCL server with provided GPU resources. YOLO models and SRGAN is run
    on Google Colab.
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# export CUDA_VISIBLE_DEVICES=1  # used for setting specific GPU in terminal
if tf.config.list_physical_devices("GPU"):
    print("Use GPU of UCL server: london.ee.ucl.ac.uk")
    physical_devices = tf.config.list_physical_devices("GPU")
    # print(physical_devices)
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    print("Use CPU of your PC.")

if __name__ == "__main__":
    """
    Notice that you can specify certain task and model for experiment by passing in
    arguments. Guidelines for running are provided in README.md and Github link.
    """
    # argument processing
    parser = argparse.ArgumentParser(description="Argparse")
    parser.add_argument(
        "--task",
        type=str,
        default="IC",
        help="choose IG for image generation and IC for image classification.",
    )
    parser.add_argument("--method", type=str, default="CNN", help="model chosen")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size of NNs like MLP and CNN"
    )
    parser.add_argument("--epochs", type=int, default=50, help="epochs of NNs")
    parser.add_argument(
        "--lr", type=float, default=0.00001, help="learning rate of NNs"
    )
    parser.add_argument(
        "--multilabel",
        type=bool,
        default=False,
        help="whether consider multilabel setting for CNN",
    )
    args = parser.parse_args()
    task = args.task
    method = args.method
    print(f"Method: {method} Task: {task} Multilabel: {args.multilabel}.")

    # load data
    print("Start loading data......")
    pre_path = (
        "datasets/pencil/" if method in ["PencilGAN"] else "datasets/preprocessed/"
    )

    if method in ["CNN"]:  # 100x100x3
        train_ds, val_ds, test_ds = load_data(
            task, pre_path, method, batch_size=args.batch_size
        )
    elif method in ["MoE", "Multimodal"]:
        train_dataset, val_dataset, test_dataset = load_data(
            task, pre_path, method, batch_size=args.batch_size
        )
    elif method in [
        "ViT",
        "ConGAN",
        "AutoEncoder",
        "AdvCNN",
        "BaseGAN",
        "PencilGAN",
        "ResNet50",
        "InceptionV3",
        "MobileNetV2",
        "NASNetMobile",
        "VGG19",
    ]:
        Xtrain, ytrain, Xtest, ytest, Xval, yval = load_data(
            task, pre_path, method, batch_size=args.batch_size
        )
    print("Load data successfully.")

    # model selection
    print("Start loading model......")
    if method in ["CNN"]:
        model = load_model(task, method, args.multilabel, args.lr)
    elif method in [
        "AutoEncoder",
        "ViT",
        "AdvCNN",
        "ConGAN",
        "PencilGAN",
        "BaseGAN",
        "MoE",
        "ResNet50",
        "InceptionV3",
        "MobileNetV2",
        "NASNetMobile",
        "VGG19",
        "Multimodal",
    ]:
        model = load_model(
            task, method, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs
        )
    print("Load model successfully.")

    """
        This part includes all training, validation and testing process with encapsulated functions.
        Detailed process of each method can be seen in corresponding classes.
    """
    if method in ["CNN"]:
        if args.multilabel == False:
            train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
                model, train_ds, val_ds, args.epochs
            )
            test_res, pred_test, ytest = model.test(model, test_ds)
        else:  # multilabel
            (
                train_res,
                val_res,
                pred_train,
                pred_train_multilabel,
                pred_val,
                pred_val_multilabel,
                ytrain,
                yval,
            ) = model.train(model, train_ds, val_ds, args.epochs)
            test_res, pred_test, pred_test_multilabel, ytest = model.test(
                model, test_ds
            )
    elif method in ["MoE", "Multimodal"]:
        pred_train, pred_val, ytrain, yval = model.train(
            train_dataset, val_dataset, test_dataset
        )
        pred_test, ytest = model.test(test_dataset)
    elif method in [
        "AdvCNN",
        "ResNet50",
        "InceptionV3",
        "MobileNetV2",
        "NASNetMobile",
        "VGG19",
    ]:
        train_res, val_res, pred_train, pred_val, ytrain, yval = model.train(
            Xtrain, ytrain, Xval, yval
        )
        pred_test, ytest = model.test(Xtest, ytest)
    elif method in ["BaseGAN", "PencilGAN"]:
        model.train(model, Xtrain)
        model.generate()
    elif method in ["ConGAN"]:
        model.train(model, Xtrain, ytrain)
        model.generate()
    elif method in ["AutoEncoder"]:
        train_res, val_res = model.train(Xtrain, ytrain, Xval, yval)
        test_res = model.test(Xtest)
    elif method == "ViT":
        train_res, val_res, pred_train, ytrain, pred_val, yval = model.train(
            Xtrain, ytrain, Xval, yval
        )
        test_res, pred_test, ytest = model.test(Xtest, ytest)

    # metrics and visualization
    if task == "IC":
        res = {
            "train_res": get_metrics(task, ytrain, pred_train),
            "val_res": get_metrics(task, yval, pred_val),
            "test_res": get_metrics(task, ytest, pred_test),
        }
        for i in res.items():
            print(i)
        if args.multilabel == True:
            method = method + "_multilabel"
        visual4cm(task, method, ytrain, yval, ytest, pred_train, pred_val, pred_test)
        if method != "MoE" and ("CNN" not in method) and method != "Multimodal":
            visual4loss(
                method, "train", train_res["train_loss"], train_res["train_acc"]
            )
            visual4loss(method, "val", val_res["val_loss"], val_res["val_acc"])
