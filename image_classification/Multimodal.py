# -*- encoding: utf-8 -*-
"""
@File    :   Multimodal.py
@Time    :   2024/02/24 22:05:30
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all procedures for multimodality model.
"""

# here put the import lib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Lambda,
)

# late fusion:
# segmentation 100x100x3
# original 100x100x3
# contour 100x100x3
# point cloud 4096x3 xyz 4096x3 rgb 4096x6 -- 64x64x6


class Multimodal(Model):
    def __init__(self, method, lr=0.001, batch_size=32, epochs=10):
        super(Multimodal, self).__init__()

        self.input_2d = Input(shape=(100, 100, 3))  # input of seg/contour/original
        self.input_3d = Input(shape=(64, 64, 6))  # input of point cloud
        self.ori_model = self.baseNet(self.input_2d)  # original
        self.seg_model = self.baseNet(self.input_2d)  # segmentation model
        self.contour_model = self.baseNet(self.input_2d)  # contour
        self.pc_model = self.baseNet(self.input_3d)  # point cloud
        self.oriModel = Model(self.input_2d, self.ori_model)
        self.segModel = Model(self.input_2d, self.seg_model)
        self.contourModel = Model(self.input_2d, self.contour_model)
        self.pcModel = Model(self.input_3d, self.pc_model)

        self.ori_output_layer = tf.keras.models.Model(
            inputs=self.oriModel.input, outputs=self.oriModel.layers[-1].output
        )
        self.seg_output_layer = tf.keras.models.Model(
            inputs=self.segModel.input, outputs=self.segModel.layers[-1].output
        )
        self.contour_output_layer = tf.keras.models.Model(
            inputs=self.contourModel.input, outputs=self.contourModel.layers[-1].output
        )
        self.pc_output_layer = tf.keras.models.Model(
            inputs=self.pcModel.input, outputs=self.pcModel.layers[-1].output
        )

        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = EarlyStopping(
            monitor="val_loss", patience=2, verbose=1, mode="auto"
        )
        self.epochs = epochs
        self.method = method

    def baseNet(self, x):
        """
        description: This method is used for building basic network of each modality
        param {*} self
        param {*} x: input
        return {*}: output
        """
        h = Conv2D(32, 3, padding="same", activation="relu")(x)
        h = BatchNormalization()(h)
        h = Conv2D(32, 3, padding="same", activation="relu")(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.3)(h)

        h = Conv2D(64, 3, padding="same", activation="relu")(h)
        h = Conv2D(64, 3, padding="same", activation="relu")(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Dropout(0.25)(h)

        h = Conv2D(128, 3, padding="same", activation="relu")(h)
        h = Conv2D(128, 3, padding="same", activation="relu")(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.3)(h)

        h = Flatten()(h)
        h = Dense(512, activation="relu")(h)
        out = Dense(12, activation="softmax")(h)  # none,12
        return out

    def train(self, train_dataset, val_dataset, test_dataset):
        """
        description: This function is used for the entire process of training.
        param {*} self
        param {*} train_dataset: loaded train dataset as batches
        param {*} val_dataset: loaded validation dataset as batches
        param {*} test_dataset: loaded test dataset as batches
        return {*}: prediction of train and validation
        """
        print("Start training......")
        train_pred, val_pred = [], []

        # general
        print("Training for RGB image classification......")
        self.oriModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.oriModel.fit(
            train_dataset[0],
            to_categorical(train_dataset[4], 12),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[0], to_categorical(val_dataset[4], 12)),
            callbacks=[self.early_stop],
        )
        ori_train_predictions = self.ori_output_layer.predict(x=train_dataset[0])
        ori_train_prob = tf.nn.softmax(ori_train_predictions)  # probabilities

        # segmentation
        print("Training for segmentation image classification......")
        self.segModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.segModel.fit(
            train_dataset[1],
            to_categorical(train_dataset[4], 12),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[1], to_categorical(val_dataset[4], 12)),
            callbacks=[self.early_stop],
        )
        seg_train_predictions = self.seg_output_layer.predict(x=train_dataset[1])
        seg_train_prob = tf.nn.softmax(seg_train_predictions)

        # contour
        print("Training for contour image classification......")
        self.contourModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.contourModel.fit(
            train_dataset[2],
            to_categorical(train_dataset[4], 12),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[2], to_categorical(val_dataset[4], 12)),
            callbacks=[self.early_stop],
        )
        contour_train_predictions = self.contour_output_layer.predict(
            x=train_dataset[2]
        )
        contour_train_prob = tf.nn.softmax(contour_train_predictions)

        # point cloud
        print("Training for point cloud image classification......")
        self.pcModel.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        self.pcModel.fit(
            train_dataset[3],
            to_categorical(train_dataset[4], 12),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(val_dataset[3], to_categorical(val_dataset[4], 12)),
            callbacks=[self.early_stop],
        )
        pc_train_predictions = self.pc_output_layer.predict(x=train_dataset[3])
        pc_train_prob = tf.nn.softmax(pc_train_predictions)

        # freeze all trained models
        for i in [self.oriModel, self.segModel, self.contourModel, self.pcModel]:
            for l in i.layers:
                l.trainable = False

        # fusion: late decision with average
        train_predictions = (
            ori_train_prob + seg_train_prob + contour_train_prob + pc_train_prob
        ) / 4
        train_prob = tf.nn.softmax(train_predictions)
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)

        # validation
        print("Evaluating for multimodal classification......")
        for model in [self.oriModel, self.segModel, self.pcModel, self.contourModel]:
            for l in model.layers:
                print(l, l.trainable)
        ori_val_predictions = self.ori_output_layer.predict(x=val_dataset[0])
        ori_val_prob = tf.nn.softmax(ori_val_predictions)
        seg_val_predictions = self.seg_output_layer.predict(x=val_dataset[1])
        seg_val_prob = tf.nn.softmax(seg_val_predictions)
        contour_val_predictions = self.contour_output_layer.predict(x=val_dataset[2])
        contour_val_prob = tf.nn.softmax(contour_val_predictions)
        pc_val_predictions = self.pc_output_layer.predict(x=val_dataset[3])
        pc_val_prob = tf.nn.softmax(pc_val_predictions)

        val_predictions = (
            ori_val_prob + seg_val_prob + contour_val_prob + pc_val_prob
        ) / 4
        val_prob = tf.nn.softmax(val_predictions)
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)
        return train_pred, val_pred, train_dataset[4], val_dataset[4]

    def test(self, test_dataset):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} test_dataset: loaded test dataset as batches
        return {*}: prediction of test
        """
        print("Start testing......")
        test_pred = []
        ori_test_predictions = self.ori_output_layer.predict(x=test_dataset[0])
        ori_test_prob = tf.nn.softmax(ori_test_predictions)
        seg_test_predictions = self.seg_output_layer.predict(x=test_dataset[1])
        seg_test_prob = tf.nn.softmax(seg_test_predictions)
        contour_test_predictions = self.contour_output_layer.predict(x=test_dataset[2])
        contour_test_prob = tf.nn.softmax(contour_test_predictions)
        pc_test_predictions = self.pc_output_layer.predict(x=test_dataset[3])
        pc_test_prob = tf.nn.softmax(pc_test_predictions)

        # fusion
        test_predictions = (
            ori_test_prob + seg_test_prob + contour_test_prob + pc_test_prob
        ) / 4
        test_prob = tf.nn.softmax(test_predictions)
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)
        print("Finish testing.")

        return test_pred, test_dataset[4]
