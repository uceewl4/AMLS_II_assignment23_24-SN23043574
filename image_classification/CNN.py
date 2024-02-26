# -*- encoding: utf-8 -*-
"""
@File    :   CNN.py
@Time    :   2024/02/24 21:47:03
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    : This file includes all implementation process of CNN and multilabel setting.
"""

# here put the import lib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorboardX import SummaryWriter  # used for nn curves visualization
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)


class CNN(Model):
    def __init__(self, method, multilabel=False, lr=0.001):
        super(CNN, self).__init__()
        self.multilabel = multilabel
        self.method = method + "_multilabel" if self.multilabel == True else method

        self.c1 = Conv2D(
            32, 3, padding="same", activation="relu", input_shape=(100, 100, 3)
        )
        self.b1 = BatchNormalization()
        self.c2 = Conv2D(32, 3, padding="same", activation="relu")
        self.b2 = BatchNormalization()
        self.p1 = MaxPooling2D()
        self.dp1 = Dropout(0.3)

        self.c3 = Conv2D(64, 3, padding="same", activation="relu")
        self.b3 = BatchNormalization()
        self.c4 = Conv2D(64, 3, padding="same", activation="relu")
        self.b4 = BatchNormalization()
        self.p2 = MaxPooling2D()
        self.dp2 = Dropout(0.2)  # 0.2

        self.fc = Flatten()
        self.d1 = Dense(512, activation="relu")
        self.d2 = Dense(64, activation="relu")
        self.d3 = Dense(12, name="outputs")  # 12-class classification

        # objective function: sparse categorical cross entropy for mutliclass classification
        # notice that here the loss is calculated from logits, no need to set activation function for the output layer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        )
        self.lr = lr
        # adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # loss and accuracy
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.val_loss = tf.keras.metrics.Mean(name="eval_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy"
        )
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

    def call(self, x):
        """
        description: This function is the actual construction process of the network
        param {*} self
        param {*} x: input
        return {*}: output logits
        """
        x = self.c1(x)
        x = self.b1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.p1(x)
        x = self.dp1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.p2(x)
        x = self.dp2(x)

        x = self.fc(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

    def train(self, model, train_ds, val_ds, EPOCHS):
        """
        description: This function is used for the entire process of training.
            Notice that loss of both train and validation are backward propagated.
        param {*} self
        param {*} model: customized network constructed
        param {*} train_ds: loaded train dataset as batches
        param {*} val_ds: loaded validation dataset as batches
        param {*} EPOCHS: number of epochs
        return {*}: results of train and validation
        """
        print("Start training......")
        if not os.path.exists("outputs/image_classification/nn_curves/"):
            os.makedirs("outputs/image_classification/nn_curves/")
        writer = SummaryWriter(f"outputs/image_classification/nn_curves/{self.method}")

        # train
        for epoch in range(EPOCHS):
            train_pred = []  # label prediction
            ytrain = []  # ground truth
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            for step, (train_images, train_labels) in enumerate(train_ds):
                with tf.GradientTape() as tape:
                    predictions = model(train_images, training=True)  # logits

                    if self.multilabel == False:
                        train_prob = tf.nn.softmax(predictions)  # probabilities
                        train_pred += np.argmax(train_prob, axis=1).tolist()
                    else:  # top-3 multilabels
                        train_prob_multilabel = tf.nn.softmax(predictions)
                        train_pred_multilabel = np.zeros_like(predictions)
                        top3 = np.argsort(train_prob_multilabel)[:, -3:][:, ::-1]
                        for index, i in enumerate(top3.tolist()):
                            train_pred_multilabel[index, i] = 1

                        # if original label belong to top-3 multilabel prediction, assume it as correct
                        # otherwise use the largest probability in multilabel prediction index
                        for index, i in enumerate(np.array(train_labels).tolist()):
                            if train_labels[index] in top3[index]:
                                train_pred.append(train_labels[index])
                            else:
                                train_pred.append(
                                    np.argmax(train_prob_multilabel[index]).tolist()
                                )
                    ytrain += np.array(train_labels).tolist()  # ground truth
                    loss = self.loss_object(train_labels, predictions)

                # backward propagation
                gradients = tape.gradient(loss, model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables)
                )
                self.train_loss(loss)
                self.train_accuracy(train_labels, predictions)

                # validation
                if step % 50 == 0:
                    val_pred, yval = [], []
                    self.val_loss.reset_states()
                    self.val_accuracy.reset_states()

                    for val_images, val_labels in val_ds:
                        with tf.GradientTape() as tape:
                            predictions = model(val_images, training=True)

                            if self.multilabel == False:
                                val_prob = tf.nn.softmax(predictions)
                                val_pred += np.argmax(val_prob, axis=1).tolist()
                            else:
                                val_prob_multilabel = tf.nn.softmax(predictions)
                                val_pred_multilabel = np.zeros_like(predictions)
                                top3 = np.argsort(val_prob_multilabel)[:, -3:][:, ::-1]
                                for index, i in enumerate(top3.tolist()):
                                    val_pred_multilabel[index, i] = 1

                                for index, i in enumerate(
                                    np.array(val_labels).tolist()
                                ):
                                    if val_labels[index] in top3[index]:
                                        val_pred.append(val_labels[index])
                                    else:
                                        val_pred.append(
                                            np.argmax(
                                                val_prob_multilabel[index]
                                            ).tolist()
                                        )
                            yval += np.array(val_labels).tolist()
                            val_loss = self.loss_object(val_labels, predictions)
                            self.val_loss(val_loss)
                            self.val_accuracy(val_labels, predictions)

                        # backward propagation
                        gradients = tape.gradient(val_loss, model.trainable_variables)
                        self.optimizer.apply_gradients(
                            zip(gradients, model.trainable_variables)
                        )
                        self.val_loss(val_loss)
                        self.val_accuracy(val_labels, predictions)

                    val_res = {
                        "val_loss": np.array(self.val_loss.result()).tolist(),
                        "val_acc": round(np.array(self.val_accuracy.result()) * 100, 4),
                    }
                    print(f"Epoch: {epoch + 1}, Step: {step} ", val_res)

            train_res = {
                "train_loss": np.array(self.train_loss.result()).tolist(),
                "train_acc": round(np.array(self.train_accuracy.result()) * 100, 4),
            }
            print(f"Epoch: {epoch + 1}", train_res)

            # tensorboardX
            writer.add_scalars(
                "loss",
                {
                    "train_loss": np.array(self.train_loss.result()).tolist(),
                    "val_loss": np.array(self.val_loss.result()).tolist(),
                },
                epoch,
            )
            writer.add_scalars(
                "accuracy",
                {
                    "train_accuracy": np.array(self.train_accuracy.result()).tolist(),
                    "val_accuracy": np.array(self.val_accuracy.result()).tolist(),
                },
                epoch,
            )

            train_pred = np.array(train_pred)
            val_pred = np.array(val_pred)

        print("Finish training.")
        writer.close()

        if self.multilabel == False:
            return train_res, val_res, train_pred, val_pred, ytrain, yval
        else:
            return (
                train_res,
                val_res,
                train_pred,
                train_pred_multilabel,
                val_pred,
                val_pred_multilabel,
                ytrain,
                yval,
            )

    def test(self, model, test_ds):
        """
        description: This function is used for the entire process of testing.
            Notice that loss of testing is not backward propagated.
        param {*} self
        param {*} model: network constructed
        param {*} test_ds: loaded test dataset as batches
        return {*}: result of testing
        """
        print("Start testing......")
        test_pred = []  # predicted labels
        ytest = []  # ground truth
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()

        for test_images, test_labels in test_ds:
            predictions = model(test_images, training=False)  # logits

            if self.multilabel == False:
                test_prob = tf.nn.softmax(predictions)
                test_pred += np.argmax(test_prob, axis=1).tolist()
            else:  # multilabel
                test_prob_multilabel = tf.nn.softmax(predictions)
                test_pred_multilabel = np.zeros_like(predictions)
                top3 = np.argsort(test_prob_multilabel)[:, -3:][:, ::-1]
                for index, i in enumerate(top3.tolist()):
                    test_pred_multilabel[index, i] = 1

                for index, i in enumerate(np.array(test_labels).tolist()):
                    if test_labels[index] in top3[index]:
                        test_pred.append(test_labels[index])
                    else:
                        test_pred.append(
                            np.argmax(test_prob_multilabel[index]).tolist()
                        )

            ytest += np.array(test_labels).tolist()
            t_loss = self.loss_object(test_labels, predictions)
            self.test_loss(t_loss)
            self.test_accuracy(test_labels, predictions)

        test_res = {
            "test_loss": np.array(self.test_loss.result()).tolist(),
            "test_acc": round(np.array(self.test_accuracy.result()) * 100, 4),
        }
        test_pred = np.array(test_pred)
        print("Finish testing.")

        if self.multilabel == False:
            return test_res, test_pred, ytest
        else:
            return test_res, test_pred, test_pred_multilabel, ytest
