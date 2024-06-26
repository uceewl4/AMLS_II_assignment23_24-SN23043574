# -*- encoding: utf-8 -*-
"""
@File    :   AdvCNN.py
@Time    :   2024/02/24 21:40:58
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all implementation process of adversarial CNN.
code refers to https://github.com/sabrinagoellner/secure-machine-learning/blob/main/adversarial_attack_MNIST.ipynb
"""


# here put the import lib
import numpy as np
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
import tensorflow as tf
from tensorflow.keras import Model, models
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential
import random
import cv2
import os


class AdvCNN(Model):
    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super(AdvCNN, self).__init__()

        self.model = Sequential(
            [
                Conv2D(
                    32, 3, padding="same", activation="relu", input_shape=(100, 100, 3)
                ),
                BatchNormalization(),
                Conv2D(32, 3, padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.3),
                Conv2D(64, 3, padding="same", activation="relu"),
                BatchNormalization(),
                Conv2D(64, 3, padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.4),
                Conv2D(128, 3, padding="same", activation="relu"),
                BatchNormalization(),
                Conv2D(128, 3, padding="same", activation="relu"),
                BatchNormalization(),
                MaxPooling2D(),
                Dropout(0.3),  # 0.2
                Flatten(),
                Dense(256, activation="relu"),
                Dense(64, activation="relu"),
                Dense(12, name="outputs"),  # 12-class
            ]
        )
        self.model.build((None, 100, 100, 3))
        self.model.summary()

        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.get_layer("outputs").output
        )  # get logits

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.classifier = TensorFlowV2Classifier(
            clip_values=(0, 1),
            model=self.model,
            nb_classes=12,
            input_shape=(100, 100, 3),
            loss_object=self.loss_object,
        )

    def train(self, Xtrain, ytrain, Xval, yval):
        """
        description: This function includes entire training and validation process for the method.
        param {*} self
        param {*} Xtrain: train images
        param {*} ytrain: train ground truth labels
        param {*} Xval: validation images
        param {*} yval: validation ground truth labels
        return {*}: train and validation results
        """

        print(f"Start training for {self.method}......")
        train_pred, val_pred = [], []  # label prediction

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_object,
            metrics=["accuracy"],
        )

        # generate adversarial images
        adv_attack = FastGradientMethod(estimator=self.classifier, eps=0.3)
        adv_index = random.sample([i for i in range(len(Xtrain))], 300)  # 300,100,100
        Xtrain_adv = adv_attack.generate(x=Xtrain[adv_index, :, :, :])
        ytrain_adv = np.array(
            [i for index, i in enumerate(ytrain.tolist()) if index in adv_index]
        )
        Xtrain = np.concatenate((Xtrain, Xtrain_adv), axis=0)
        ytrain = np.concatenate((ytrain, ytrain_adv), axis=0)
        # save adversarial images
        if not os.path.exists("outputs/image_classification/adversarial/train"):
            os.makedirs("outputs/image_classification/adversarial/train")
        for index, img in enumerate(Xtrain_adv.tolist()):
            cv2.imwrite(
                f"outputs/image_classification/adversarial/train/{ytrain_adv[index]}_{index}.JPG",
                np.array(img),
            )

        adv_index = random.sample([i for i in range(len(Xval))], 100)
        Xval_adv = adv_attack.generate(x=Xval[adv_index, :, :, :])
        yval_adv = np.array(
            [i for index, i in enumerate(yval.tolist()) if index in adv_index]
        )
        Xval = np.concatenate((Xval, Xval_adv), axis=0)
        yval = np.concatenate((yval, yval_adv), axis=0)
        if not os.path.exists("outputs/image_classification/adversarial/val"):
            os.makedirs("outputs/image_classification/adversarial/val")
        for index, img in enumerate(Xval_adv.tolist()):
            cv2.imwrite(
                f"outputs/image_classification/adversarial/val/{yval_adv[index]}_{index}.JPG",
                np.array(img),
            )

        history = self.model.fit(
            Xtrain,
            ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(Xval, yval),
        )

        # get predictions
        train_predictions = self.output_layer.predict(x=Xtrain)
        train_prob = tf.nn.softmax(train_predictions)  # probabilities
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_predictions = self.output_layer.predict(x=Xval)
        val_prob = tf.nn.softmax(val_predictions)  # probabilities
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)
        val_res = {
            "val_loss": history.history["val_loss"],
            "val_acc": history.history["val_accuracy"],
        }

        print(f"Finish training for {self.method}.")
        return train_res, val_res, train_pred, val_pred, ytrain, yval

    def test(self, Xtest, ytest):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} Xtest: test images
        param {*} ytest: test ground truth labels
        return {*}: test results
        """
        print("Start testing......")
        test_pred = []

        # adversarial attack
        adv_index = random.sample([i for i in range(len(Xtest))], 100)
        adv_attack = ProjectedGradientDescent(
            self.classifier,
            eps=0.3,
            eps_step=0.01,
            max_iter=40,
            targeted=False,
            num_random_init=True,
        )
        Xtest_adv = adv_attack.generate(x=Xtest[adv_index, :, :, :])
        ytest_adv = np.array(
            [i for index, i in enumerate(ytest.tolist()) if index in adv_index]
        )
        Xtest = np.concatenate((Xtest, Xtest_adv), axis=0)
        ytest = np.concatenate((ytest, ytest_adv), axis=0)
        # save adversarial
        if not os.path.exists("outputs/image_classification/adversarial/test"):
            os.makedirs("outputs/image_classification/adversarial/test")
        for index, img in enumerate(Xtest_adv.tolist()):
            cv2.imwrite(
                f"outputs/image_classification/adversarial/test/{ytest_adv[index]}_{index}.JPG",
                np.array(img),
            )

        # get predictions
        test_loss, test_acc = self.model.evaluate(Xtest, ytest, verbose=2)
        test_predictions = self.output_layer.predict(x=Xtest)
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)

        print("Finish training.")
        return ytest, test_pred
