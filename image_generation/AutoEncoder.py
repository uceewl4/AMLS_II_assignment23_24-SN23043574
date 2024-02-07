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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dense, Embedding, Concatenate, LayerNormalization, Add, Dropout, MultiHeadAttention,Flatten,Reshape
from tensorflow.keras.models import Model
import cv2
import random
import os 


class AutoEncoder(Model):

    """
    description: This function is used for initialization of Inception-V3 + classifiers.
    param {*} self
    param {*} method: baseline model selected
    """

    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super(AutoEncoder, self).__init__()

        self.encoder_input = Input(shape=(100, 100, 1))
        self.h1 = Flatten()(self.encoder_input)
        self.h2 = Dense(5000, activation="relu")(self.h1)
        self.h3 = Dense(1000, activation="relu")(self.h2)
        self.encoder_output = Dense(25*25, activation="relu")(self.h3)
        self.encoder = Model(self.encoder_input, self.encoder_output)

        self.decoder_input = Dense(25*25, activation="relu")(self.encoder_output)
        self.h4 = Dense(1000, activation="relu")(self.decoder_input)
        self.h5 = Dense(5000, activation="relu")(self.h4)
        self.h6 = Dense(10000, activation="relu")(self.h5)
        self.decoder_output = Reshape((100, 100, 1))(self.h6)

        self.model = Model(self.encoder_input, self.decoder_output)  # autoencoder
        self.model.build((None,100,100,1))
        self.model.summary()

        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
        )

        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
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
        print(Xtrain)
        self.model.compile(
            optimizer=self.optimizer,
            loss='mse',
        )

        history = self.model.fit(
            Xtrain,ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(Xval,yval),
            # callbacks=[self.early_stop],
        )
        self.model.save("outputs/image_generation/models/autoEncoder.h5")

        train_generated_images = self.model.predict(Xtrain)*255
        # print(train_generated_images)
        train_index = random.sample([i for i in range(len(Xtrain))],300)
        if not os.path.exists("outputs/image_generation/autoEncoder/train"):
            os.makedirs("outputs/image_generation/autoEncoder/train")
        for index,i in enumerate(train_index):
            cv2.imwrite(f'outputs/image_generation/autoEncoder/train/img_{i}.JPG',train_generated_images[index])

        train_res = {
            "val_loss": history.history["val_loss"],
        }
        
        val_generated_images = self.model.predict(Xval)*255
        val_index = random.sample([i for i in range(len(Xval))],100)
        if not os.path.exists("outputs/image_generation/autoEncoder/val"):
            os.makedirs("outputs/image_generation/autoEncoder/val")
        for index in val_index:
            cv2.imwrite(f'outputs/image_generation/autoEncoder/val/img_{index}.JPG',val_generated_images[index])
        val_res = {
            "val_loss": history.history["val_loss"],
        }
        
        print(f"Finish training for {self.method}.")
        # result is used for drawing curves
        return train_res, val_res

        

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} ytrain: train ground truth labels
    param {*} yval: validation ground truth labels
    return {*}: predicted labels for train, validation and test respectively
  """

    def test(self, Xtest):
        print("Start testing......")
        self.model.load_weights("outputs/image_generation/models/autoEncoder.h5")
        test_generated_images = self.model.predict(Xtest)*255
        test_index = random.sample([i for i in range(len(Xtest))],100)
        if not os.path.exists("outputs/image_generation/autoEncoder/test"):
            os.makedirs("outputs/image_generation/autoEncoder/test")
        for index in test_index:
            cv2.imwrite(f'outputs/image_generation/autoEncoder/test/img_{index}.JPG',test_generated_images[index])

        test_res = {
            "test_loss": self.model.evaluate(Xtest)
        }
        print(f"Finish testing. Test loss: {test_res['test_loss']}")
        return test_res

    

