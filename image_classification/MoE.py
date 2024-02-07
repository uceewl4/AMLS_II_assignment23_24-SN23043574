"""
Author: uceewl4 uceewl4@ucl.ac.uk
Date: 2024-01-30 19:38:07
LastEditors: uceewl4 uceewl4@ucl.ac.uk
LastEditTime: 2024-01-30 19:41:19
FilePath: /AMLS_II_assignment23_24-SN23043574/image_classification/MoE.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""

# split into furniture and home goods
# look at the part misclassified
# three experts: furniture/home goods/misclassified samples

# here put the import lib
import numpy as np
import tensorflow as tf
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
    Lambda,
    Reshape
)
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
import numpy as np


class MoE(Model):
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
        super(MoE, self).__init__()

        self.orig_class = 12
        self.gate_class = 3  # trinary classification for expert
        self.inputs = Input(shape=(100,100,3), name="input")
        self.base_model = self.baseNet(self.inputs,self.orig_class)
        self.gate_model = self.baseNet(self.inputs,self.gate_class)
        self.fur_model = self.baseNet(self.inputs,self.orig_class)
        self.good_model = self.baseNet(self.inputs,self.orig_class)
        self.mis_model = self.baseNet(self.inputs,self.orig_class)
        self.baseModel = Model(self.inputs, self.base_model)
        self.gateModel = Model(self.inputs, self.gate_model)
        self.furModel= Model(self.inputs, self.fur_model)
        self.goodModel = Model(self.inputs, self.good_model)
        self.misModel = Model(self.inputs, self.mis_model)
        
        self.furGate = self.subGate(self.inputs)
        self.goodGate = self.subGate(self.inputs)
        self.misGate = self.subGate(self.inputs)
        # 0 fur 1 good 2 mis
        # self.lambda_function1 = lambda gx: self.select(gx)
        # self.merge = Lambda(self.lambda_function1,output_shape=(self.orig_class,))
        # self.outputs = self.merge([self.base_model, self.gate_model, self.fur_model, self.good_model, self.mis_model,self.furGate,self.goodGate,self.misGate])
        self.outputs = Lambda(lambda gx: self.select(gx),output_shape=(self.orig_class,))([self.base_model, self.gate_model, self.fur_model, self.good_model, self.mis_model,self.furGate,self.goodGate,self.misGate])
        self.model = Model(self.inputs,self.outputs)
        self.output_layer = tf.keras.models.Model(
            inputs=self.model.input, outputs=self.model.layers[-1].output
        )

        self.lr = lr
        self.batch_size = batch_size
        # self.early_stop = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
        self.epochs = epochs

        # adam optimizer
        self.method = method

    """
    description: This function is the actual construction process of customized network.
    param {*} self
    param {*} x: input 
    return {*}: output logits
    """

    def baseNet(self, x,num_classes):
        h = Conv2D(32,3, padding='same', activation='relu')(x)
        h = BatchNormalization()(h)
        h = Conv2D(32, 3, padding='same', activation='relu')(h)
        h = BatchNormalization()(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.3)(h)
        h = Conv2D(64, 3, padding='same', activation='relu')(h)
        h = Conv2D(128, 3, padding='same', activation='relu')(h)
        h = MaxPooling2D(pool_size=(2,2))(h)
        h = Dropout(0.25)(h)
        h = Conv2D(128, 3, padding='same', activation='relu')(h)
        h = Conv2D(256, 3, padding='same', activation='relu')(h)
        h = Conv2D(256, 3, padding='same', activation='relu')(h)
        h = MaxPooling2D()(h)
        h = Dropout(0.2)(h)

        h = Flatten()(h)
        h = Dense(512, activation='relu')(h)
        h = Dense(128, activation='relu')(h)
        out = Dense(num_classes, activation='softmax')(h)
        return out

    # define sub-Gate network, for the second gating network layer
    def subGate(self,x):
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.orig_class*2, activation='softmax')(x)
        out = Reshape((self.orig_class, 2))(x)
        return out
    
    def combine(self,x):
        return tf.multiply(x[0],x[2][:,:,0]) + tf.multiply(x[1],x[2][:,:,1])
     
        # return x[0]*(x[2][:,:,0]) + x[1]*(x[2][:,:,1])
    def subGateLambda(self,base, expert, subgate):
        # self.lambda_function2 = lambda gx: self.combine(gx)
        # self.sub_function = Lambda(self.lambda_function2,output_shape=(self.orig_class,))
        # output = self.sub_function([base, expert, subgate])
        self.sub_function = Lambda(lambda gx: self.combine(gx),output_shape=(self.orig_class,))
        output = self.sub_function([base, expert, subgate])
        return output
    
    @tf.function
    def select(self,gx):
        # return tf.cond((tf.expand_dims(gx[1][:,0],axis=1) < tf.expand_dims(gx[1][:,1],axis=1)), \
        #         lambda: tf.cond((tf.expand_dims(gx[1][:,1],axis=1) < tf.expand_dims(gx[1][:,2],axis=1)),lambda: self.subGateLambda(gx[0], gx[4], gx[7]),lambda: self.subGateLambda(gx[0], gx[3], gx[6])), \
        #         lambda: tf.cond((tf.expand_dims(gx[1][:,0],axis=1) < tf.expand_dims(gx[1][:,2],axis=1)),lambda: self.subGateLambda(gx[0], gx[4], gx[7]),lambda: self.subGateLambda(gx[0], gx[2], gx[5])))
        return tf.where((tf.expand_dims(gx[1][:,0],axis=1) < tf.expand_dims(gx[1][:,1],axis=1)), \
                  tf.where((tf.expand_dims(gx[1][:,1],axis=1) < tf.expand_dims(gx[1][:,2],axis=1)),\
                           self.subGateLambda(gx[0], gx[4], gx[7]), self.subGateLambda(gx[0], gx[3], gx[6])), \
                  tf.where((tf.expand_dims(gx[1][:,0],axis=1) < tf.expand_dims(gx[1][:,2],axis=1)),\
                           self.subGateLambda(gx[0], gx[4], gx[7]), self.subGateLambda(gx[0], gx[2], gx[5])))



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

    def train(self, train_dataset, val_dataset, test_dataset):

        print("Start training......")
        print("Training for base model of 12 class classification......")
        train_pred, val_pred = [],[]
        self.baseModel.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                   metrics=['accuracy'])

        # X, XFur, XGood, XMis, \ y, yExp, yFur, yGood, yMis
        self.baseModel.fit(train_dataset[0], train_dataset[4],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[0], val_dataset[4]))
        
        # intermediate evaluate for test set
        base_test_loss, base_test_accuracy = self.baseModel.evaluate(test_dataset[0], test_dataset[4])
        print(f"Pre-training for base model: loss: {base_test_loss}, acc: {base_test_accuracy}")

        print("Training for gate classifier of 3 class classification......")
        self.gateModel.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                   metrics=['accuracy'])
        self.gateModel.fit(train_dataset[0], train_dataset[5],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[0], val_dataset[5]))
        gate_test_loss, gate_test_accuracy = self.gateModel.evaluate(test_dataset[0], test_dataset[5])
        print(f"Pre-training for gate model: loss: {gate_test_loss}, acc: {gate_test_accuracy}")

        print("Training for furniture classifier of 3 class......")
        self.furModel.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                   metrics=['accuracy'])

        # X, XFur, XGood, XMis, \ y, yExp, yFur, yGood, yMis
        self.furModel.fit(train_dataset[1], train_dataset[6],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[1], val_dataset[6]))
        fur_test_loss, fur_test_accuracy = self.furModel.evaluate(test_dataset[1], test_dataset[6])
        print(f"Pre-training for furniture model: loss: {fur_test_loss}, acc: {fur_test_accuracy}")

        print("Training for good classifier of 7 class......")
        self.goodModel.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                   metrics=['accuracy'])

        # X, XFur, XGood, XMis, \ y, yExp, yFur, yGood, yMis
        self.goodModel.fit(train_dataset[2], train_dataset[7],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[2], val_dataset[7]))
        good_test_loss, good_test_accuracy = self.goodModel.evaluate(test_dataset[2], test_dataset[7])
        print(f"Pre-training for good model: loss: {good_test_loss}, acc: {good_test_accuracy}")

        print("Training for mis classifier of 2 class......")
        self.misModel.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                   metrics=['accuracy'])

        # X, XFur, XGood, XMis, \ y, yExp, yFur, yGood, yMis
        self.misModel.fit(train_dataset[3], train_dataset[8],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[3], val_dataset[8]))
        mis_test_loss, mis_test_accuracy = self.misModel.evaluate(test_dataset[3], test_dataset[8])
        print(f"Pre-training for mis model: loss: {mis_test_loss}, acc: {mis_test_accuracy}")
        
        for i in [self.baseModel,self.gateModel,self.furModel,self.goodModel,self.misModel]:
            for l in i.layers:
                l.trainable = False
        

        print("Training for gate and importance......")
        self.model.compile(loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                   metrics=['accuracy'])
        for l in self.model.layers:
            print(l, l.trainable)
        # print(train_dataset[0].shape)
        # print(train_dataset[4].shape)
        self.model.fit(train_dataset[0], train_dataset[4],
               batch_size=self.batch_size,
               epochs=self.epochs,
               validation_data=(val_dataset[0], val_dataset[4]))
        
        train_predictions = self.output_layer.predict(x=train_dataset[0])
        train_prob = tf.nn.softmax(train_predictions)  # probabilities
        train_pred += np.argmax(train_prob, axis=1).tolist()
        train_pred = np.array(train_pred)

        val_predictions = self.output_layer.predict(x=val_dataset[0])
        val_prob = tf.nn.softmax(val_predictions)  # probabilities
        val_pred += np.argmax(val_prob, axis=1).tolist()
        val_pred = np.array(val_pred)

        ytrain = np.argmax(train_dataset[4], axis=1)
        yval = np.argmax(val_dataset[4], axis=1)
        # print(ytrain)
        # print(ytrain.shape)
        # print(ytrain.flatten())
        
        return train_pred, val_pred, ytrain,yval


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
        moe_test_loss, moe_test_accuracy = self.model.evaluate(test_dataset[0], test_dataset[4])
        print(f"Testing for MoE model: loss: {moe_test_loss}, acc: {moe_test_accuracy}")
        test_predictions = self.output_layer.predict(x=test_dataset[0])
        test_prob = tf.nn.softmax(test_predictions)  # probabilities
        # print(np.argmax(test_prob, axis=1).tolist())
        test_pred += np.argmax(test_prob, axis=1).tolist()
        # print(test_pred)
        test_pred = np.array(test_pred)
        ytest = np.argmax(test_dataset[4], axis=1)
        

        print("Finish testing.")
        
        return test_pred, ytest
        

