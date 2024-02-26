# -*- encoding: utf-8 -*-
"""
@File    :   ViT.py
@Time    :   2024/02/24 22:15:03
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all procedure for vision transformer.
The code refers to https://github.com/nikhilroxtomar/Flower-Image-Classification-using-Vision-Transformer/blob/main/vit.py
"""

# here put the import lib
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, models
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Concatenate,
    LayerNormalization,
    Add,
    Dropout,
    MultiHeadAttention,
    Layer,
)
from tensorflow.keras.models import Model


# this class is used for creating extra class token for all patches
class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls


class ViT(Model):
    def __init__(self, method=None, lr=0.00001, epochs=10, batch_size=32):
        super(ViT, self).__init__()

        self.num_patches = 100  # 100x100x3 -- 100 patches
        self.patch_size = 10  # 10x10x3
        self.channel = 3
        self.hidden_dim = 768
        self.num_layers = 12  # num of layers for transformer
        self.mlp_dim = 300  # num of dimension for MLP
        self.num_heads = 12  # multihead
        self.inputs = Input((100, 10 * 10 * 3))  # (None,100 patches,300)

        # patch + position embedding
        self.patch_embed = Dense(self.hidden_dim)(
            self.inputs
        )  # (None, 100, 768)  dimension for embedding
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)
        self.pos_embed = Embedding(
            input_dim=self.num_patches, output_dim=self.hidden_dim
        )(
            self.positions
        )  # (100, 768)
        self.embed = self.patch_embed + self.pos_embed  # (None, 100, 768)

        # class token
        self.token = ClassToken()(
            self.embed
        )  # the individual one for all patches to reduce bias
        self.hidden = Concatenate(axis=1)(
            [self.token, self.embed]
        )  # (None, 100+1, 768)

        # transformer block
        for _ in range(self.num_layers):  # 101,768
            self.hidden = self.transformer_encoder(self.hidden)

        # classification head
        self.hidden = LayerNormalization(epsilon=1e-7)(self.hidden)  # (None, 101, 768)
        self.hidden = self.hidden[:, 0, :]  # select the class token
        self.outputs = Dense(12, activation="softmax")(self.hidden)

        self.model = Model(self.inputs, self.outputs)
        self.model.build((None, 100, 300))
        self.model.summary()
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
        )

        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    def mlp(self, x):
        """
        description: This method defines the MLP used in transformer encoder.
        param {*} self
        param {*} x: input
        return {*}: output
        """
        x = Dense(self.mlp_dim, activation="gelu")(x)
        x = Dropout(0.1)(x)
        x = Dense(self.hidden_dim)(x)
        x = Dropout(0.1)(x)
        return x

    def transformer_encoder(self, x):
        """
        description: This method defines basic transformer encoder block.
        param {*} self
        param {*} x: input
        return {*}: output
        """
        skip_1 = x
        x = LayerNormalization(epsilon=1e-7)(x)
        x = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.hidden_dim)(x, x)
        x = Add()([x, skip_1])  # resiual connection

        skip_2 = x
        x = LayerNormalization(epsilon=1e-7)(x)
        x = self.mlp(x)
        x = Add()([x, skip_2])

        return x

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
        train_prob = self.model.predict(x=Xtrain)  # softmax
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)
        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }

        val_prob = self.model.predict(x=Xtrain)  # softmax
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)
        val_res = {
            "val_loss": history.history["val_loss"],
            "val_acc": history.history["val_accuracy"],
        }

        print(f"Finish training for {self.method}.")
        return train_res, val_res, train_pred, ytrain, val_pred, yval

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
        test_res = self.model.evaluate(x=Xtest, verbose=2)
        test_prob = self.output_layer.predict(Xtest)
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)
        print("Finish testing.")

        return test_res, test_pred, ytest
