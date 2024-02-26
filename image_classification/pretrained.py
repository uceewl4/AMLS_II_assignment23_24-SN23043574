# -*- encoding: utf-8 -*-
"""
@File    :   pretrained.py
@Time    :   2024/02/24 22:11:47
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :  This file includes all procedures of pretrained models for image classification.
Five pretrained model can be selected including InceptionV3, ResNet50, VGG19, MobileNetV2, NASNetMobile.
"""

# here put the import lib

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, models


class Pretrained(Model):
    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super(Pretrained, self).__init__()

        # pretrained model
        if method == "InceptionV3":
            self.base_model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_shape=(100, 100, 3),
            )
        elif method == "ResNet50":
            self.base_model = tf.keras.applications.ResNet50(
                include_top=False, weights="imagenet", input_shape=(100, 100, 3)
            )
        elif method == "VGG19":
            self.base_model = tf.keras.applications.VGG19(
                include_top=False,
                weights="imagenet",
                input_shape=(100, 100, 3),
            )
        elif method == "MobileNetV2":
            self.base_model = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=(100, 100, 3),
            )
        elif method == "NASNetMobile":
            self.base_model = tf.keras.applications.NASNetMobile(
                include_top=False,
                weights="imagenet",
                input_shape=(100, 100, 3),
            )
        self.base_model.trainable = False
        self.model = models.Sequential(
            [
                self.base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(12, name="outputs"),  # 12-class classification
            ]
        )
        self.model.build((None, 100, 100, 3))
        self.model.summary()
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.get_layer("outputs").output
        )

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

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
        history = self.model.fit(
            Xtrain,
            ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(Xval, yval),
        )

        # get predictions
        train_predictions = self.output_layer.predict(x=Xtrain)
        train_prob = tf.nn.softmax(train_predictions)
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_predictions = self.output_layer.predict(x=Xval)
        val_prob = tf.nn.softmax(val_predictions)
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
        test_loss, test_acc = self.model.evaluate(Xtest, ytest, verbose=2)
        test_predictions = self.output_layer.predict(x=Xtest)
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)
        print("Finish training.")

        return ytest, test_pred
