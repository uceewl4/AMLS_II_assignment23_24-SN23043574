# -*- encoding: utf-8 -*-
"""
@File    :   MoE.py
@Time    :   2024/02/24 21:53:36
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :  This file includes all procedures of multi-agent.
The code refers to AMLS II lab 1 MoE.
"""

# here put the import lib

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Lambda,
    Reshape,
)
import tensorflow.keras.backend as K
import numpy as np


class MoE(Model):
    def __init__(self, method, lr=0.001, batch_size=32, epochs=10):
        super(MoE, self).__init__()

        self.orig_class = 12
        self.gate_class = 3
        self.inputs = Input(shape=(100, 100, 3), name="input")
        self.base_model = self.baseNet(
            self.inputs, self.orig_class
        )  # general 12-class classifier
        self.gate_model = self.baseNet(
            self.inputs, self.gate_class
        )  # triple classification for expert
        self.fur_model = self.baseNet(
            self.inputs, self.orig_class
        )  # furniture classifier (3)
        self.good_model = self.baseNet(
            self.inputs, self.orig_class
        )  # home goods classifier (7)
        self.mis_model = self.baseNet(self.inputs, self.orig_class)  # misclassified (2)
        self.baseModel = Model(self.inputs, self.base_model)
        self.gateModel = Model(self.inputs, self.gate_model)
        self.furModel = Model(self.inputs, self.fur_model)
        self.goodModel = Model(self.inputs, self.good_model)
        self.misModel = Model(self.inputs, self.mis_model)

        # corresponding gate
        self.furGate = self.subGate(self.inputs)
        self.goodGate = self.subGate(self.inputs)
        self.misGate = self.subGate(self.inputs)
        # 0 fur 1 good 2 mis

        self.outputs = Lambda(
            lambda gx: self.select(gx), output_shape=(self.orig_class,)
        )(
            [
                self.base_model,
                self.gate_model,
                self.fur_model,
                self.good_model,
                self.mis_model,
                self.furGate,
                self.goodGate,
                self.misGate,
            ]
        )
        self.model = Model(self.inputs, self.outputs)
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
        )

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.method = method

    def baseNet(self, x, num_classes):
        """
        description: this methods is used for building basic network architecture.
        param {*} self
        param {*} x: input
        param {*} num_classes: num of classes
        return {*}: output
        """
        h = Conv2D(32, 3, padding="same", activation="relu")(x)
        h = BatchNormalization()(h)
        h = Conv2D(32, 3, padding="same", activation="relu")(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.3)(h)

        h = Conv2D(64, 3, padding="same", activation="relu")(h)
        h = Conv2D(128, 3, padding="same", activation="relu")(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Dropout(0.25)(h)

        h = Conv2D(128, 3, padding="same", activation="relu")(h)
        h = Conv2D(256, 3, padding="same", activation="relu")(h)
        h = Conv2D(256, 3, padding="same", activation="relu")(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.2)(h)

        h = Flatten()(h)
        h = Dense(512, activation="relu")(h)
        h = Dense(128, activation="relu")(h)
        out = Dense(num_classes, activation="softmax")(h)
        return out

    # define sub-Gate network, for the second gating network layer

    def subGate(self, x):
        """
        description: This method is used for constructing sub-gate to determine significance.
        param {*} self
        param {*} x: input
        return {*}: output
        """
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(self.orig_class * 2, activation="softmax")(x)
        out = Reshape((self.orig_class, 2))(x)  # significance for general and expert
        return out

    def combine(self, x):
        """
        description: This method is used for getting final prediction as a combination of general and expert
        with corresponding significance.
        param {*} self
        param {*} x: input
        return {*}: output
        """
        return tf.multiply(x[0], x[2][:, :, 0]) + tf.multiply(x[1], x[2][:, :, 1])

    def subGateLambda(self, base, expert, subgate):
        self.sub_function = Lambda(
            lambda gx: self.combine(gx), output_shape=(self.orig_class,)
        )
        output = self.sub_function([base, expert, subgate])
        return output

    @tf.function
    def select(self, gx):
        """
        description: This function is used for selecting from three experts.
        return {*}: selected result
        """
        return tf.where(
            (tf.expand_dims(gx[1][:, 0], axis=1) < tf.expand_dims(gx[1][:, 1], axis=1)),
            tf.where(
                (
                    tf.expand_dims(gx[1][:, 1], axis=1)
                    < tf.expand_dims(gx[1][:, 2], axis=1)
                ),
                self.subGateLambda(gx[0], gx[4], gx[7]),
                self.subGateLambda(gx[0], gx[3], gx[6]),
            ),
            tf.where(
                (
                    tf.expand_dims(gx[1][:, 0], axis=1)
                    < tf.expand_dims(gx[1][:, 2], axis=1)
                ),
                self.subGateLambda(gx[0], gx[4], gx[7]),
                self.subGateLambda(gx[0], gx[2], gx[5]),
            ),
        )

    def train(self, train_dataset, val_dataset, test_dataset):
        """
        description: This function is used for the entire process of training and validation.
        param {*} self
        param {*} train_dataset: loaded train dataset as batches
        param {*} val_dataset: loaded validation dataset as batches
        param {*} test_dataset: loaded test dataset as batches
        return {*}: prediction of train and validation
        """

        print("Start training......")
        train_pred, val_pred = [], []

        # general
        print("Training for base model of 12 class classification......")
        self.baseModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        # X, XFur, XGood, XMis, \ y, yExp, yFur, yGood, yMis
        self.baseModel.fit(
            train_dataset[0],
            train_dataset[4],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[0], val_dataset[4]),
        )
        # intermediate evaluate for test set
        base_test_loss, base_test_accuracy = self.baseModel.evaluate(
            test_dataset[0], test_dataset[4]
        )
        print(
            f"Pre-training for base model: loss: {base_test_loss}, acc: {base_test_accuracy}"
        )

        # triple gate
        print("Training for gate classifier of 3 class classification......")
        self.gateModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.gateModel.fit(
            train_dataset[0],
            train_dataset[5],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[0], val_dataset[5]),
        )
        gate_test_loss, gate_test_accuracy = self.gateModel.evaluate(
            test_dataset[0], test_dataset[5]
        )
        print(
            f"Pre-training for gate model: loss: {gate_test_loss}, acc: {gate_test_accuracy}"
        )

        # furniture
        print("Training for furniture classifier of 3 class......")
        self.furModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.furModel.fit(
            train_dataset[1],
            train_dataset[6],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[1], val_dataset[6]),
        )
        fur_test_loss, fur_test_accuracy = self.furModel.evaluate(
            test_dataset[1], test_dataset[6]
        )
        print(
            f"Pre-training for furniture model: loss: {fur_test_loss}, acc: {fur_test_accuracy}"
        )

        # home goods
        print("Training for good classifier of 7 class......")
        self.goodModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.goodModel.fit(
            train_dataset[2],
            train_dataset[7],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[2], val_dataset[7]),
        )
        good_test_loss, good_test_accuracy = self.goodModel.evaluate(
            test_dataset[2], test_dataset[7]
        )
        print(
            f"Pre-training for good model: loss: {good_test_loss}, acc: {good_test_accuracy}"
        )

        # misclassified
        print("Training for mis classifier of 2 class......")
        self.misModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.misModel.fit(
            train_dataset[3],
            train_dataset[8],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[3], val_dataset[8]),
        )
        mis_test_loss, mis_test_accuracy = self.misModel.evaluate(
            test_dataset[3], test_dataset[8]
        )
        print(
            f"Pre-training for mis model: loss: {mis_test_loss}, acc: {mis_test_accuracy}"
        )

        # freeze trained model
        for i in [
            self.baseModel,
            self.gateModel,
            self.furModel,
            self.goodModel,
            self.misModel,
        ]:
            for l in i.layers:
                l.trainable = False

        # significance
        print("Training for gate and importance......")
        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        for l in self.model.layers:
            print(l, l.trainable)
        self.model.fit(
            train_dataset[0],
            train_dataset[4],
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[0], val_dataset[4]),
        )

        # get predictions
        train_predictions = self.output_layer.predict(x=train_dataset[0])
        train_prob = tf.nn.softmax(train_predictions)  # probabilities
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)

        val_predictions = self.output_layer.predict(x=val_dataset[0])
        val_prob = tf.nn.softmax(val_predictions)
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)

        ytrain = np.argmax(train_dataset[4], axis=1)
        yval = np.argmax(val_dataset[4], axis=1)

        return train_pred, val_pred, ytrain, yval

    def test(self, test_dataset):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} test_dataset: loaded test dataset as batches
        return {*}: prediction of test
        """
        print("Start testing......")
        test_pred = []
        moe_test_loss, moe_test_accuracy = self.model.evaluate(
            test_dataset[0], test_dataset[4]
        )
        print(f"Testing for MoE model: loss: {moe_test_loss}, acc: {moe_test_accuracy}")
        test_predictions = self.output_layer.predict(x=test_dataset[0])
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)
        ytest = np.argmax(test_dataset[4], axis=1)
        print("Finish testing.")

        return test_pred, ytest
