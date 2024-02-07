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
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, LayerNormalization, Add, Dropout, MultiHeadAttention, Layer
from tensorflow.keras.models import Model


class ClassToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = w_init(shape=(1, 1, input_shape[-1]), dtype=tf.float32),
            trainable = True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        hidden_dim = self.w.shape[-1]

        cls = tf.broadcast_to(self.w, [batch_size, 1, hidden_dim])
        cls = tf.cast(cls, dtype=inputs.dtype)
        return cls
    
class ViT(Model):

    """
    description: This function is used for initialization of Inception-V3 + classifiers.
    param {*} self
    param {*} method: baseline model selected
    """

    def __init__(self, method=None, lr=0.00001, epochs=10, batch_size=32):
        super(ViT, self).__init__()

        self.num_patches = 100  # 100/10 ** 2
        self.patch_size = 10  # 100,100,3
        self.channel = 3
        self.hidden_dim = 768  # 
        self.num_layers = 12  # num of layers for transformer
        self.mlp_dim = 300
        self.num_heads = 12
        # self.input_shape = (self.num_patches, self.patch_size*self.patch_size*self.channel)
        
        self.inputs = Input((100,10*10*3))     ## (None, 256, 3072) (None,100,300)

        """ Patch + Position Embeddings """
        self.patch_embed = Dense(self.hidden_dim)(self.inputs)   ## (None, 256, 768)  dimension for embedding
        self.positions = tf.range(start=0, limit=self.num_patches, delta=1)
        self.pos_embed = Embedding(input_dim=self.num_patches, output_dim=self.hidden_dim)(self.positions) ## (256, 768)
        self.embed = self.patch_embed + self.pos_embed ## (None, 256, 768)  None,100,768

        """ Adding Class Token """
        self.token = ClassToken()(self.embed)  # the individual one for all patches to reduce bias
        self.hidden = Concatenate(axis=1)([self.token, self.embed]) ## (None, 257, 768)  101,768

        for _ in range(self.num_layers):  # 257,768
            self.hidden = self.transformer_encoder(self.hidden)

        """ Classification Head """
        self.hidden = LayerNormalization(epsilon=1e-7)(self.hidden)     ## (None, 257, 768)  101, 768
        self.hidden = self.hidden[:, 0, :]  # 768, select the global sentence one
        self.outputs= Dense(12, activation="softmax")(self.hidden)  # probabilities

        self.model = Model(self.inputs, self.outputs)

        self.model.build((None,100,300))
        self.model.summary()

        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
        )

        # print(self.model.layers[-1].output)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.lr = lr
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)


    def mlp(self, x):
        x = Dense(self.mlp_dim, activation="gelu")(x)
        x = Dropout(0.1)(x)
        x = Dense(self.hidden_dim)(x)
        x = Dropout(0.1)(x)
        return x

    def transformer_encoder(self,x):
        skip_1 = x
        x = LayerNormalization(epsilon=1e-7)(x)
        x = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.hidden_dim
        )(x, x)
        x = Add()([x, skip_1])
        # x = x + skip_1

        skip_2 = x
        x = LayerNormalization(epsilon=1e-7)(x)
        x = self.mlp(x)
        x = Add()([x, skip_2])
        # x = x+skip_2

        return x
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

    def train(self,Xtrain,ytrain,Xval,yval):
        # concate with classifier
        print(f"Start training for {self.method}......")
        train_pred, val_pred = [], []  # label prediction

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss_object,
            metrics=["accuracy"],
        )

        history = self.model.fit(
            Xtrain,ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_data=(Xval,yval),
        )

        train_prob = self.model.predict(x=Xtrain)  # softmax
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)

        train_res = {
            "train_loss": history.history["loss"],
            "train_acc": history.history["accuracy"],
        }  

        val_res = {
            "val_loss": history.history["val_loss"],
            "val_acc": history.history["val_accuracy"],
        }

        val_prob = self.model.predict(x=Xtrain)  # softmax
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)

        print(f"Finish training for {self.method}.")
        # result is used for drawing curves

        return train_res, val_res, train_pred, ytrain, val_pred, yval
    
    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} ytrain: train ground truth labels
    param {*} yval: validation ground truth labels
    return {*}: predicted labels for train, validation and test respectively
  """

    def test(self, Xtest,ytest):
        print("Start testing......")
        test_pred = []
        test_res = self.model.evaluate(x=Xtest, verbose=2)
        test_prob = self.output_layer.predict(Xtest)  # softmax
        test_pred += np.argmax(test_prob, axis=1).tolist()
        test_pred = np.array(test_pred)

        print("Finish testing.")

        return test_res, test_pred, ytest
