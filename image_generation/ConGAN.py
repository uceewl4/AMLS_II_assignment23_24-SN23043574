# -*- encoding: utf-8 -*-
"""
@File    :   ConGAN.py
@Time    :   2024/02/24 22:54:34
@Programme :  MSc Integrated Machine Learning Systems (TMSIMLSSYS01)
@Module : ELEC0135: Applied Machine Learning Systems II
@SN :   23043574
@Contact :   uceewl4@ucl.ac.uk
@Desc    :   This file includes all procedures for conditional GAN.
The code refers to https://github.com/nicknochnack/GANBasics.
"""

# here put the import lib
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Flatten,
    Reshape,
    LeakyReLU,
    Dropout,
    UpSampling2D,
    Input,
    Embedding,
    Concatenate,
)
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback


# this class is used for periodically generating images to monitor performance.
class ModelMonitor(Callback):
    def __init__(self, num_img=3, latent_dim=200):  # 200 samples, 3 fake images
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = np.random.randn(3, self.latent_dim)
        random_latent_label = np.random.randint(0, 12, 3)
        generated_images = self.model.generator(
            [random_latent_vectors, random_latent_label]
        )
        generated_images *= 255
        generated_images.numpy()
        if not os.path.exists("outputs/image_generation/conGAN"):
            os.makedirs("outputs/image_generation/conGAN")
        for i in range(self.num_img):
            img = array_to_img(generated_images[i])
            if (epoch in range(0, 800, 10)) or (epoch in range(0, 10)):
                img.save(
                    f"outputs/image_generation/conGAN/generated_img_{epoch}_{i}.png"
                )
            else:
                pass


class ConGAN(Model):
    def __init__(self, method=None, lr=0.001, epochs=10, batch_size=32):
        super().__init__()

        self.generator = self.genNet()  # generator
        self.discriminator = self.disNet()  # discriminator
        self.generator.summary()
        self.discriminator.summary()
        self.epoch = epochs
        self.batch_size = batch_size
        self.method = method
        self.lr = lr

    def genNet(self):
        """
        description: This method is used for building architecture of generator.
        return {*}: generator model
        """
        # random label
        label = Input(shape=(1,))
        embedding = Embedding(12, 50)(label)  # (1, 50)  12 classes
        hidden = Dense(25 * 25)(embedding)  # label features
        hidden = Reshape((25, 25, 1))(hidden)

        # random input
        input = Input(shape=(200,))
        gen = Dense(25 * 25 * 200)(input)  # begin with 200 samples into 25*25*200
        gen = LeakyReLU(0.2)(gen)
        gen = Reshape((25, 25, 200))(gen)
        gen = Concatenate()([gen, hidden])

        # upsampling block
        gen = UpSampling2D()(gen)  # 50,50,64
        gen = Conv2D(64, 5, padding="same")(gen)
        gen = LeakyReLU(0.2)(gen)

        gen = UpSampling2D()(gen)  # 100,100,128
        gen = Conv2D(128, 5, padding="same")(gen)
        gen = LeakyReLU(0.2)(gen)

        # convolutional block
        gen = Conv2D(256, 4, padding="same")(gen)
        gen = LeakyReLU(0.2)(gen)

        gen = Conv2D(256, 4, padding="same")(gen)
        gen = LeakyReLU(0.2)(gen)

        output = Conv2D(1, 4, padding="same", activation="sigmoid")(gen)  # 1 channel
        model = Model([input, label], output)

        return model

    def disNet(self):
        """
        description: This methods is used for building discriminator architecture.
        return {*}: discriminator
        """
        # input label
        label = Input(shape=(1,))
        embedding = Embedding(12, 50)(label)
        embedding = Dense(100 * 100)(embedding)
        hidden = Reshape((100, 100, 1))(embedding)

        # input images
        input = Input(shape=(100, 100, 1))
        merge = Concatenate()([input, hidden])

        # convolutional block
        dis = Conv2D(32, 5, input_shape=(100, 100, 2))(merge)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.4)(dis)

        dis = Conv2D(64, 5)(dis)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.3)(dis)

        dis = Conv2D(128, 5)(dis)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.3)(dis)

        dis = Conv2D(256, 5)(dis)
        dis = LeakyReLU(0.2)(dis)
        dis = Dropout(0.3)(dis)

        dis = Flatten()(dis)
        dis = Dropout(0.3)(dis)
        output = Dense(1, activation="sigmoid")(dis)
        model = Model([input, label], output)

        return model

    def compile(self, gen_optimizer, dis_optimizer, gen_loss, dis_loss):
        super().compile()
        self.gen_optimizer = gen_optimizer
        self.dis_optimizer = dis_optimizer
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss

    def train_step(self, train_batch):
        """
        description: This method is used  to override the train_step method in fit
        param {*} self
        param {*} train_batch
        return {*}: result of loss
        """
        # fake/real images/labels
        real_image = train_batch[0]
        real_label = train_batch[1]
        fake_label = np.random.randint(0, 12, self.batch_size)
        fake_image = self.generator(
            [np.random.randn(self.batch_size, 200), fake_label], training=False
        )  # fake images from 200 random samples

        # train discriminator
        with tf.GradientTape() as dis_tape:
            # pass concatenation of images and labels
            pred_real = self.discriminator([real_image, real_label], training=True)
            pred_fake = self.discriminator([fake_image, fake_label], training=True)
            pred_total = tf.concat([pred_real, pred_fake], axis=0)
            ground_total = tf.concat(
                [tf.zeros_like(pred_real), tf.ones_like(pred_fake)], axis=0
            )  # zero for real, one for fake
            # add random noise
            noise_real = 0.2 * tf.random.uniform(tf.shape(pred_real))
            noise_fake = -0.2 * tf.random.uniform(tf.shape(pred_fake))
            pred_total += tf.concat([noise_real, noise_fake], axis=0)
            # binary crossentropy loss
            dis_loss = self.dis_loss(ground_total, pred_total)

        # discriminator backward propagation
        dis_grad = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)
        self.dis_optimizer.apply_gradients(
            zip(dis_grad, self.discriminator.trainable_variables)
        )

        # train generator
        with tf.GradientTape() as gen_tape:
            fake_label = np.random.randint(0, 12, self.batch_size)
            fake_image = self.generator(
                [np.random.randn(self.batch_size, 200), fake_label], training=False
            )  # fake images from 64 random samples
            pred_fake_gen = self.discriminator([fake_image, fake_label], training=False)
            # binary crossentropy loss
            gen_loss = self.gen_loss(tf.zeros_like(pred_fake_gen), pred_fake_gen)

        # generator backward propagation
        gen_grad = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.gen_optimizer.apply_gradients(
            zip(gen_grad, self.generator.trainable_variables)
        )
        return {"dis_loss": dis_loss, "gen_loss": gen_loss}

    def train(self, model, Xtrain, ytrain):
        """
        description: This function includes entire training process for the method.
        param {*} self
        param {*} model: constructed GAN
        param {*} Xtrain: train images
        param {*} ytrain: train real labels
        """
        print(f"Start training for {self.method}......")
        model.compile(
            tf.keras.optimizers.Adam(learning_rate=self.lr),
            tf.keras.optimizers.Adam(learning_rate=self.lr),
            tf.keras.losses.BinaryCrossentropy(),
            tf.keras.losses.BinaryCrossentropy(),
        )
        history = model.fit(
            Xtrain,
            ytrain,
            batch_size=self.batch_size,
            epochs=self.epoch,
            callbacks=[ModelMonitor()],
        )

        # save models
        if not os.path.exists("outputs/image_generation/models/"):
            os.makedirs("outputs/image_generation/models/")
        self.generator.save("outputs/image_generation/models/conGAN_generator.h5")
        self.discriminator.save(
            "outputs/image_generation/models/conGAN_disciriminator.h5"
        )

    def generate(self):
        """
        description: This function is used for generating images to evaluate image quality of GAN.
        """
        print("Start generating images with conditional GAN......")
        self.generator.load_weights(
            "outputs/image_generation/models/conGAN_generator.h5"
        )
        generated_images = (
            self.generator.predict(
                [np.random.randn(100, 200), np.random.randint(0, 12, 100)]
            )
            * 255
        )  # generate 100 images
        print(generated_images)
        if not os.path.exists("outputs/image_generation/conGAN/"):
            os.makedirs("outputs/image_generation/conGAN/")
        for index, image in enumerate(generated_images):
            cv2.imwrite(f"outputs/image_generation/conGAN/img_{index}.JPG", image)
        print("Finish generating images.")
