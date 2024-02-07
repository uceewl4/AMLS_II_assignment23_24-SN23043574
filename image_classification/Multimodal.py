# -*- encoding: utf-8 -*-
"""
@File    :   CNN.py
@Time    :   2023/12/16 21:53:45
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0134 Applied Machine Learning Systems
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file is used for customized network of CNN, including network initialization.
    construction and entire process of training, validation and testing.
"""

# here put the import lib
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Model
from tensorboardX import SummaryWriter  # used for nn curves visualization
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Lambda
)

# late fusion: 100x100x3, 100x100x3, 100x100x3, 4096x3 xyz 4096x3 rgb  4096x6 -- 124x124x6

class Multimodal(Model):
    """
    description: This function includes all initialization of CNN, like layers used for construction,
      loss function object, optimizer, measurement of accuracy and loss.
    param {*} self
    param {*} task: task A or B
    param {*} method: CNN
    param {*} multilabel: whether under multilabel setting
    param {*} lr: learning rate
    """

    def __init__(self, method, lr=0.001, batch_size=32, epochs = 10):
        super(Multimodal, self).__init__()
        # network layers definition
        self.input_2d = Input(shape=(100,100,3))
        self.input_3d = Input(shape=(124,124,6))
        self.ori_model = self.baseNet(self.input_2d)
        self.seg_model = self.baseNet(self.input_2d)
        self.contour_model = self.baseNet(self.input_2d)
        self.pc_model = self.baseNet(self.input_3d)
        self.oriModel = Model(self.input_2d, self.ori_model)
        self.segModel = Model(self.input_2d, self.seg_model)
        self.contourModel= Model(self.input_2d, self.contour_model)
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
        self. pc_output_layer = tf.keras.models.Model(
            inputs=self.pcModel.input, outputs=self.pcModel.layers[-1].output
        )


        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        self.epochs = epochs

        # adam optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.method = method

    def baseNet(self, x):
        h = Conv2D(32,3, padding='same', activation='relu')(x)
        h = BatchNormalization(h)
        h = Conv2D(32, 3, padding='same', activation='relu')(h)
        h = BatchNormalization(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.3)(h)
        h = Conv2D(64, 3, padding='same', activation='relu')(h)
        h = Conv2D(64, 3, padding='same', activation='relu')(h)
        h = MaxPooling2D(pool_size=(2,2))(h)
        h = Dropout(0.25)(h)
        h = Conv2D(128, 3, padding='same', activation='relu')(h)
        h = Conv2D(128, 3, padding='same', activation='relu')(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.3)(h)

        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        out = Dense(12, activation='softmax')(h)
        return out

    """
  description: This function is used for the entire process of training. 
    Notice that loss of both train and validation are backward propagated.
  param {*} self
  param {*} model: customized network constructed
  param {*} train_ds: loaded train dataset as batches
  param {*} val_ds: loaded validation dataset as batches
  param {*} EPOCHS: number of epochs
  return {*}: accuracy and loss results, predicted labels, ground truth labels of train and validation
  """
    # X, seg, contour, pc, y
    def train(self, train_dataset, val_dataset, test_dataset):

        print("Start training......")
        train_pred, val_pred = [], []
        print("Training for RGB image classification......")
        self.oriModel.compile(loss='categorical_crossentropy',
                   optimizer=self.optimizer,
                   metrics=['accuracy'])

        self.oriModel.fit(train_dataset[0], train_dataset[4],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[0], val_dataset[4]),
               callbacks=[self.early_stop])
        
        ori_train_predictions = self.ori_output_layer.predict(x=self.train_dataset[0])
        ori_train_prob = tf.nn.softmax(ori_train_predictions)  # probabilities

        print("Training for segmentation image classification......")
        self.segModel.compile(loss='categorical_crossentropy',
                   optimizer=self.optimizer,
                   metrics=['accuracy'])
        self.segModel.fit(train_dataset[1], train_dataset[4],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[1], val_dataset[4]),
               callbacks=[self.early_stop])
    
        seg_train_predictions = self.seg_output_layer.predict(x=self.train_dataset[1])
        seg_train_prob = tf.nn.softmax(seg_train_predictions)  # probabilities
       

        print("Training for contour image classification......")
        self.contourModel.compile(loss='categorical_crossentropy',
                   optimizer=self.optimizer,
                   metrics=['accuracy'])
        self.contourModel.fit(train_dataset[2], train_dataset[4],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[2], val_dataset[4]),
               callbacks=[self.early_stop])
    
        contour_train_predictions = self.contour_output_layer.predict(x=train_dataset[2])
        contour_train_prob = tf.nn.softmax(contour_train_predictions)  # probabilities
       
        print("Training for point cloud image classification......")
        self.pcModel.compile(loss='categorical_crossentropy',
                   optimizer=self.optimizer,
                   metrics=['accuracy'])
        self.pcModel.fit(train_dataset[3], train_dataset[4],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[3], val_dataset[4]),
               callbacks=[self.early_stop])
        pc_train_predictions = self.pc_output_layer.predict(x=train_dataset[3])
        pc_train_prob = tf.nn.softmax(pc_train_predictions)  # probabilities
       
        
        for i in [self.oriModel,self.segModel,self.contourModel,self.pcModel]:
            for l in i.layers:
                l.trainable = False

        # fusion
        train_predictions = np.mean((ori_train_prob,seg_train_prob,contour_train_prob,pc_train_prob),axis=1)
        train_prob = tf.nn.softmax(train_predictions)  # probabilities
        train_pred += np.argmax(train_prob, axis=1).tolist()

        print("Evaluating for multimodal classification......")
        for l in self.model.layers:
            print(l, l.trainable)
        ori_val_predictions = self.ori_output_layer.predict(x=val_dataset[0])
        ori_val_prob = tf.nn.softmax(ori_val_predictions)  # probabilities
        seg_val_predictions = self.seg_output_layer.predict(x=val_dataset[1])
        seg_val_prob = tf.nn.softmax(seg_val_predictions)  # probabilities
        contour_val_predictions = self.contour_output_layer.predict(x=val_dataset[2])
        contour_val_prob = tf.nn.softmax(contour_val_predictions)  # probabilities
        pc_val_predictions = self.pc_output_layer.predict(x=val_dataset[3])
        pc_val_prob = tf.nn.softmax(pc_val_predictions)  # probabilities

        val_predictions = np.mean((ori_val_prob,seg_val_prob,contour_val_prob,pc_val_prob),axis=1)
        val_prob = tf.nn.softmax(val_predictions)  # probabilities
        val_pred += np.argmax(val_prob, axis=1).tolist()
        
        return train_pred, val_pred, self.train_dataset[4], self.val_dataset[4]


    """
    description: This function is used for the entire process of testing. 
        Notice that loss of testing is not backward propagated.
    param {*} self
    param {*} model: customized network constructed
    param {*} test_ds: loaded test dataset as batches
    return {*}: accuracy and loss result, predicted labels (multilabel if necessary) and ground truth of test dataset
    """

    def test(self, test_dataset):
        print("Start testing......")
        test_pred = []
        ori_test_predictions = self.ori_output_layer.predict(x=test_dataset[0])
        ori_test_prob = tf.nn.softmax(ori_test_predictions)  # probabilities
        seg_test_predictions = self.seg_output_layer.predict(x=test_dataset[1])
        seg_test_prob = tf.nn.softmax(seg_test_predictions)  # probabilities
        contour_test_predictions = self.contour_output_layer.predict(x=test_dataset[2])
        contour_test_prob = tf.nn.softmax(contour_test_predictions)  # probabilities
        pc_test_predictions = self.pc_output_layer.predict(x=test_dataset[3])
        pc_test_prob = tf.nn.softmax(pc_test_predictions)  # probabilities

        test_predictions = np.mean((ori_test_prob,seg_test_prob,contour_test_prob,pc_test_prob),axis=1)
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        test_pred += np.argmax(test_prob, axis=1).tolist()
        print("Finish testing.")
        
        return test_pred, test_dataset[4]
        

