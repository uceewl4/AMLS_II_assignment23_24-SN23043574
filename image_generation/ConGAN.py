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
import os
import cv2
import numpy as np
from sklearn import svm
import tensorflow as tf
from tensorflow.keras import Model, models
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
# Bring in the sequential api for the generator and discriminator
from tensorflow.keras.models import Sequential
# Bring in the layers for the neural network
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D, Input, Embedding, Concatenate
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback


class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=256):  # 128 samples, 3 fake images
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = np.random.randn(3, self.latent_dim)
        random_latent_label = np.random.randint(0,12,3)
        generated_images = self.model.generator([random_latent_vectors,random_latent_label])
        generated_images *= 255
        generated_images.numpy()
        if not os.path.exists("outputs/image_generation/"):
            os.makedirs("outputs/image_generation/")
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            if epoch in range(10):
                img.save(f"outputs/image_generation/generated_img_{epoch}_{i}.png")
            else:
                pass

class ConGAN(Model):

    """
    description: This function is used for initialization of Inception-V3 + classifiers.
    param {*} self
    param {*} method: baseline model selected
    """

    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super().__init__()

        self.generator = self.genNet()
        self.discriminator = self.disNet()
        self.generator.summary()
        self.discriminator.summary()

        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.lr = lr

    def genNet(self):  # input 100 100 3
        # 100,100,3
        label = Input(shape=(1,))
        embedding = Embedding(12, 50)(label) #Shape 1,50  12 classes
        hidden = Dense(25*25)(embedding)  # start from 25*25 to 100*100
        hidden = Reshape((25,25,1))(hidden)

        input = Input(shape=(256,))

        # Takes in random values and reshapes it to 7x7x128
        # Beginnings of a generated image
        gen = Dense(25*25*256)(input) # begin with 128 samples into 7*7*128
        gen = LeakyReLU(0.2)(gen)
        gen = Reshape((25,25,256))(gen)
        gen = Concatenate()([gen,hidden])
        
        # Upsampling block 1 
        gen = UpSampling2D()(gen) # double spatial. 50 50 128
        gen = Conv2D(128, 5, padding='same')(gen)
        gen = LeakyReLU(0.2)(gen)
        
        # Upsampling block 2 
        gen = UpSampling2D()(gen) # 100 100 128 一直到需要的图片大小  
        gen = Conv2D(128, 5, padding='same')(gen)
        gen = LeakyReLU(0.2)(gen)
        
        # Convolutional block 1
        gen = Conv2D(128, 4, padding='same')(gen)
        gen = LeakyReLU(0.2)(gen)
        
        # Convolutional block 2
        gen = Conv2D(128, 4, padding='same')(gen)
        gen = LeakyReLU(0.2)(gen)
        
        # Conv layer to get to one channel
        output = Conv2D(1, 4, padding='same', activation='sigmoid')(gen) # 1 channel
        model = Model([input,label],output)

        return model
    
    def disNet(self): 
        label = Input(shape=(1,))
        embedding = Embedding(12, 50)(label) #Shape 1,50
        embedding = Dense(100*100)(embedding)  #Shape = 1, 1024
        # reshape to additional channel
        hidden = Reshape((100, 100, 1))(embedding)  #32x32x1

        input = Input(shape=(100,100,1))
        merge = Concatenate()([input, hidden])
       
        # First Conv Block
        dis = Conv2D(32, 5, input_shape = (100,100,2))(merge)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.4)(dis)
        
        # Second Conv Block
        dis = Conv2D(64, 5)(dis)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.3)(dis)
        
        # Third Conv Block
        dis = Conv2D(128, 5)(dis)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.3)(dis)
        
        # Fourth Conv Block
        dis = Conv2D(256, 5)(dis)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.3)(dis)
        
        # Flatten then pass to dense layer
        dis = Flatten()(dis)
        dis = Dropout(0.3)(dis)
        output = Dense(1, activation='sigmoid')(dis)
        model = Model([input, label], output)
        
        return model 
    
    def compile(self,gen_optimizer, dis_optimizer,gen_loss, dis_loss):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss

    def train_step(self,train_batch):  # override the train_step method in fit
        # Get the data 
        real_image = train_batch[0]
        real_label = train_batch[1]
        fake_label = np.random.randint(0,12,self.batch_size)
        fake_image = self.generator([np.random.randn(self.batch_size, 256),\
                                    fake_label], training=False)
        # batch size is 128
        # Train the discriminator
        with tf.GradientTape() as dis_tape: 
            # Pass the real and fake images to the discriminator model
            pred_real = self.discriminator([real_image,real_label], training=True) 
            pred_fake = self.discriminator([fake_image,fake_label], training=True)
            pred_total = tf.concat([pred_real, pred_fake], axis=0)
            
            # Create labels for real and fakes images
            # zero real one fake
            ground_total = tf.concat([tf.zeros_like(pred_real), tf.ones_like(pred_fake)], axis=0)
            
            # Add some noise to the TRUE outputs
            noise_real = 0.2*tf.random.uniform(tf.shape(pred_real))
            noise_fake = -0.2*tf.random.uniform(tf.shape(pred_fake))
            pred_total += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            dis_loss = self.dis_loss(ground_total, pred_total)
            
        # Apply backpropagation - nn learn 
        dis_grad = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables) 
        self.dis_optimizer.apply_gradients(zip(dis_grad, self.discriminator.trainable_variables))
        
        # Train the generator 
        with tf.GradientTape() as gen_tape: 
            # Generate some new images
            fake_label = np.random.randint(0,12,self.batch_size)
            fake_image = self.generator([np.random.randn(self.batch_size, 256),\
                                        fake_label], training=False)
                                        
            # Create the predicted labels
            pred_fake_gen = self.discriminator([fake_image,fake_label], training=False)
                                        
            # Calculate loss - trick to training to fake out the discriminator 
            # difference between real label
            gen_loss = self.gen_loss(tf.zeros_like(pred_fake_gen), pred_fake_gen) 
            
        # Apply backprop
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(gen_grad, self.generator.trainable_variables))
        
        return {"dis_loss":dis_loss, "gen_loss":gen_loss}

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

    def train(self, model, Xtrain, ytrain):
        print(f"Start training for {self.method}......")
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.lr),\
                    tf.keras.optimizers.Adam(learning_rate=self.lr),\
                    tf.keras.losses.BinaryCrossentropy(),\
                    tf.keras.losses.BinaryCrossentropy())
        history = model.fit(
            Xtrain,ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            callbacks=[ModelMonitor()]
        )
        
        if not os.path.exists("outputs/image_generation/models/"):
            os.makedirs("outputs/image_generation/models/")
        self.generator.save('outputs/image_generation/models/conGAN_generator.h5')
        self.discriminator.save('outputs/image_generation/models/conGAN_disciriminator.h5')
    

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} ytrain: train ground truth labels
    param {*} yval: validation ground truth labels
    return {*}: predicted labels for train, validation and test respectively
  """

    def generate(self):
        print("Start generating images with conditional GAN......")
        self.generator.load_weights('outputs/image_generation/models/conGAN_generator.h5')
        generated_images = self.generator.predict([np.random.randn(100,256),np.random.randint(0,12,100)])*255  # generate 100 images
        print(generated_images)
        if not os.path.exists("outputs/image_generation/conGAN/"):
            os.makedirs("outputs/image_generation/conGAN/")
        for index,image in enumerate(generated_images):
            cv2.imwrite(f'outputs/image_generation/conGAN/img_{index}.JPG',image)
        print("Finish generating images.")


