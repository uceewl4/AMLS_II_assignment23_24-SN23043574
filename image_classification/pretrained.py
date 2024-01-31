# -*- encoding: utf-8 -*-
"""
@File    :   InceptionV3.py
@Time    :   2023/12/16 22:17:47
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for pretrained model Inception-V3 as feature extractor,
  followed by 7 classifiers of ML baselines.
"""

# here put the import lib
import numpy as np
from sklearn import svm
import tensorflow as tf
from tensorflow.keras import Model, models
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Pretrained(Model):

    """
    description: This function is used for initialization of Inception-V3 + classifiers.
    param {*} self
    param {*} method: baseline model selected
    """

    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super(Pretrained, self).__init__()

        # need resizing to satisfy the minimum image size need of Inception-V3
        # 图片大小先放着看看跑出来效果怎么样
        self.data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.Resizing(75, 75, interpolation="bilinear"),
                tf.keras.layers.Rescaling(1.0 / 255, input_shape=(75, 75, 3)),
            ]
        )

        # 所有pretrained放在一起选择
        # pretrained model
        if method == "InceptionV3":
            self.base_model = tf.keras.applications.InceptionV3(
                include_top=False,
                weights="imagenet",
                input_shape=(75, 75, 3),  # 这里可能不需要改，因为图片足够大
            )
        elif method == "ResNet50":
            self.base_model = tf.keras.applications.ResNet50(
                include_top=False, weights="imagenet", input_shape=(32, 32, 3)
            )
        elif method == "VGG19":
            self.base_model = tf.keras.applications.VGG19(
                include_top=False,
                weights="imagenet",
                input_shape=(75, 75, 3),  # 这里可能不需要改，因为图片足够大
            )
        elif method == "MobileNetV2":
            self.base_model = tf.keras.applications.MobileNetV2(
                include_top=False,
                weights="imagenet",
                input_shape=(75, 75, 3),  # 这里可能不需要改，因为图片足够大
            )
        elif method == "NASNetMobile":
            self.base_model = tf.keras.applications.NASNetMobile(
                include_top=False,
                weights="imagenet",
                input_shape=(75, 75, 3),  # 这里可能不需要改，因为图片足够大
            )
        self.base_model.trainable = False

        # 接fc/dense的到输出
        self.model = models.Sequential(
            [
                self.data_augmentation,
                self.base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(12, name="outputs"),  # 12-class classification
            ]
        )
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

        # adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.early_stop = EarlyStopping(
            monitor="val_loss", patience=2, verbose=1, mode="auto"
        )

    """
    description: This function includes entire training process and
        the cross-validation procedure for baselines of KNN, DT, RF and ABC.
        Notice that because of large size of task B dataset, high dimensional features and 
        principle of some models, the entire implementation process may be extremely slow.
        It can even take several hours for a model to run. 
        Some quick models are recommended on README.md and Github link.
    param {*} self
    param {*} Xtrain: train images
    param {*} ytrain: train ground truth labels
    param {*} Xval: validation images
    param {*} yval: validation ground truth labels
    param {*} Xtest: test images
    param {*} gridSearch: whether grid search cross-validation (only for KNN, DT, RF and ABC)
    return {*}: if grid search is performed, the cv results are returned.
  """

    def train(self, Xtrain, ytrain, Xval, yval):
        # concate with classifier
        print(f"Start training for {self.method}......")
        train_pred, val_pred = [], []  # label prediction

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_object,
            metrics=["sparse_categorical_accuracy"],
        )

        history = self.model.fit(
            Xtrain,
            ytrain,
            batch_size=self.batch,
            epochs=self.epoch,
            validation_data=(Xval, yval),
            verbose=0,
            callbacks=[self.early_stop],
        )

        train_predictions = self.output_layer.predict(x=Xtrain)
        print(train_predictions.shape)
        print(train_predictions[0])

        train_prob = tf.nn.softmax(train_predictions)  # probabilities
        train_pred += np.argmax(train_prob, axis=1).tolist()

        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_predictions = self.output_layer.predict(x=Xval)
        val_prob = tf.nn.softmax(val_predictions)  # probabilities
        val_pred += np.argmax(val_prob, axis=1).tolist()

        val_res = {
            "val_loss": history.history["val_loss"],
            "val_acc": history.history["val_accuracy"],
        }

        print(f"Finish training for {self.method}.")
        # result is used for drawing curves

        return train_res, val_res, train_pred, val_pred, ytrain, yval

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} ytrain: train ground truth labels
    param {*} yval: validation ground truth labels
    return {*}: predicted labels for train, validation and test respectively
  """

    def test(self, Xtest, ytest):
        print("Start testing......")
        test_pred = []
        test_loss, test_acc = self.model.evaluate(Xtest, ytest, verbose=2)
        test_predictions = self.output_layer.predict(x=Xtest)
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()

        print("Finish training.")

        return ytest, test_pred
