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
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=128):  # 128 samples, 3 fake images
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.uniform((self.num_img, self.latent_dim,1))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        if not os.path.exists("Outputs/image_generation/"):
            os.makedirs("Outputs/image_generation/")
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            img.save(f"Outputs/image_generation/generated_img_{epoch}_{i}.png")

class PencilGAN(Model):

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

    def genNet(self):
        # 100,100,3
        model = Sequential()
        # Takes in random values and reshapes it to 7x7x128
        # Beginnings of a generated image
        model.add(Dense(25*25*128, input_dim=128)) # begin with 128 samples into 7*7*128
        model.add(LeakyReLU(0.2))
        model.add(Reshape((25,25,128)))
        
        # Upsampling block 1 
        model.add(UpSampling2D()) # double spatial. 50 50 128
        model.add(Conv2D(128, 5, padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Upsampling block 2 
        model.add(UpSampling2D()) # 100 100 128 一直到需要的图片大小  
        model.add(Conv2D(128, 5, padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Convolutional block 1
        model.add(Conv2D(128, 4, padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Convolutional block 2
        model.add(Conv2D(128, 4, padding='same'))
        model.add(LeakyReLU(0.2))
        
        # Conv layer to get to one channel
        model.add(Conv2D(1, 4, padding='same', activation='sigmoid')) # 3 channel
        
        return model
    
    def disNet(self): 
        model = Sequential()
        
        # First Conv Block
        model.add(Conv2D(32, 5, input_shape = (100,100,1)))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.4))
        
        # Second Conv Block
        model.add(Conv2D(64, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Third Conv Block
        model.add(Conv2D(128, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Fourth Conv Block
        model.add(Conv2D(256, 5))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        
        # Flatten then pass to dense layer
        model.add(Flatten())
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        return model 
    
    def compile(self,gen_optimizer, dis_optimizer,gen_loss, dis_loss):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss

    def train_step(self, train_batch):  # override the train_step method in fit
        # Get the data 
        real_image = train_batch
        fake_image = self.generator(tf.random.normal((self.batch_size, 128, 1)), training=False)
        # batch size is 128
        # Train the discriminator
        with tf.GradientTape() as dis_tape: 
            # Pass the real and fake images to the discriminator model
            pred_real = self.discriminator(real_image, training=True) 
            pred_fake = self.discriminator(fake_image, training=True)
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
            fake_image = self.generator(tf.random.normal((self.batch_size,128,1)), training=True)
                                        
            # Create the predicted labels
            pred_fake_gen = self.discriminator(fake_image, training=False)
                                        
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

    def train(self, model, Xtrain):
        print(f"Start training for {self.method}......")
        model.compile(tf.keras.optimizers.Adam(learning_rate=self.lr),\
                    tf.keras.optimizers.Adam(learning_rate=self.lr),\
                    tf.keras.losses.BinaryCrossentropy(),\
                    tf.keras.losses.BinaryCrossentropy())
        history = model.fit(
            Xtrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            callbacks=[ModelMonitor()]
        )
        
        if not os.path.exists("Outputs/image_generation/models/"):
            os.makedirs("Outputs/image_generation/models/")
        self.generator.save('Outputs/image_generation/models/pencilGAN_generator.h5')
        self.discriminator.save('Outputs/image_generation/models/pecilGAN_disciriminator.h5')
    

    """
    description: This function is used for the entire process of testing.
    param {*} self
    param {*} ytrain: train ground truth labels
    param {*} yval: validation ground truth labels
    return {*}: predicted labels for train, validation and test respectively
  """

    def generate(self):
        print("Start generating images with pencil GAN......")
        self.generator.load_weights('Outputs/image_generation/models/pencilGAN_generator.h5')
        generated_images = self.generator.predict(tf.random.normal((100,128,1)))
        for index,image in enumerate(generated_images):
            cv2.imwrite(f'Outputs/image_generation/pencilGAN/img_{index}',image)
        print("Finish generating images.")


