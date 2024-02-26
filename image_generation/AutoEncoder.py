# -*- encoding: utf-8 -*-
"""
@File    :   AutoEncoder.py
@Time    :   2024/02/24 22:35:54
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all procedures for autoencoder.
"""
# here put the import lib
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Reshape,
)
from tensorflow.keras.models import Model
import cv2
import random
import os


class AutoEncoder(Model):
    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super(AutoEncoder, self).__init__()

        # encoder
        self.encoder_input = Input(shape=(100, 100, 1))  # original images
        self.h1 = Flatten()(self.encoder_input)
        self.h2 = Dense(2000, activation="relu")(self.h1)
        self.h3 = Dense(25 * 25, activation="relu")(self.h2)
        self.encoder_output = Dense(5 * 5, activation="relu")(self.h3)
        self.encoder = Model(self.encoder_input, self.encoder_output)  # 25

        # decoder
        self.decoder_input = Dense(5 * 5, activation="relu")(self.encoder_output)
        self.h4 = Dense(25 * 25, activation="relu")(self.decoder_input)
        self.h5 = Dense(2000, activation="relu")(self.h4)
        self.h6 = Dense(10000, activation="relu")(self.h5)
        self.decoder_output = Reshape((100, 100, 1))(self.h6)  # reconstructed images

        self.model = Model(self.encoder_input, self.decoder_output)
        self.model.build((None, 100, 100, 1))
        self.model.summary()
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
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

        self.model.compile(
            optimizer=self.optimizer,
            loss="mse",
        )
        history = self.model.fit(
            Xtrain,
            ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(Xval, yval),
        )
        self.model.save("outputs/image_generation/models/autoEncoder.h5")

        # train generated images
        train_generated_images = self.model.predict(Xtrain) * 255
        train_index = random.sample([i for i in range(len(Xtrain))], 300)
        if not os.path.exists("outputs/image_generation/autoEncoder/train"):
            os.makedirs("outputs/image_generation/autoEncoder/train")
        for index, i in enumerate(train_index):
            cv2.imwrite(
                f"outputs/image_generation/autoEncoder/train/img_{index}.JPG",
                train_generated_images[index],
            )
        train_res = {
            "val_loss": history.history["val_loss"],
        }

        # validation generated images
        val_generated_images = self.model.predict(Xval) * 255
        val_index = random.sample([i for i in range(len(Xval))], 100)
        if not os.path.exists("outputs/image_generation/autoEncoder/val"):
            os.makedirs("outputs/image_generation/autoEncoder/val")
        for index in val_index:
            cv2.imwrite(
                f"outputs/image_generation/autoEncoder/val/img_{index}.JPG",
                val_generated_images[index],
            )
        val_res = {
            "val_loss": history.history["val_loss"],
        }

        print(f"Finish training for {self.method}.")
        return train_res, val_res

    def test(self, Xtest):
        """
        description: This function is used for the entire process of testing.
        param {*} self
        param {*} Xtest: test images
        return {*}: test results
        """
        print("Start testing......")
        self.model.load_weights("outputs/image_generation/models/autoEncoder.h5")
        test_generated_images = self.model.predict(Xtest) * 255
        test_index = random.sample([i for i in range(len(Xtest))], 100)
        if not os.path.exists("outputs/image_generation/autoEncoder/test"):
            os.makedirs("outputs/image_generation/autoEncoder/test")
        for index in test_index:
            cv2.imwrite(
                f"outputs/image_generation/autoEncoder/test/img_{index}.JPG",
                test_generated_images[index],
            )
        test_res = {"test_loss": self.model.evaluate(Xtest)}
        print(f"Finish testing. Test loss: {test_res['test_loss']}")
        return test_res
