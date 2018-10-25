"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import pickle
from collections import defaultdict

import keras.backend as K
K.set_image_dim_ordering('th')

import keras
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os
import sys
import re
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout

from keras.layers import multiply as kmultiply
from keras.layers import add as kadd

import csv

from PIL import Image

from utils import save_image_array


class BalancingGAN:
    def build_generator(self, latent_size, init_resolution=8):
        resolution = self.resolution
        channels = self.channels

        # we will map a pair of (z, L), where z is a latent vector and L is a
        # label drawn from P_c, to image space (..., 3, resolution, resolution)
        cnn = Sequential()

        cnn.add(Dense(1024, input_dim=latent_size, activation='relu', use_bias=False))
        cnn.add(Dense(128 * init_resolution * init_resolution, activation='relu', use_bias=False))
        cnn.add(Reshape((128, init_resolution, init_resolution)))
        crt_res = init_resolution

        # upsample
        while crt_res != resolution:
            cnn.add(UpSampling2D(size=(2, 2)))
            if crt_res < resolution/2:
                cnn.add(Conv2D(
                    256, (5, 5), padding='same',
                    activation='relu', kernel_initializer='glorot_normal', use_bias=False)
                )

            else:
                cnn.add(Conv2D(128, (5, 5), padding='same',
                                      activation='relu', kernel_initializer='glorot_normal', use_bias=False))

            crt_res = crt_res * 2
            assert crt_res <= resolution,\
                "Error: final resolution [{}] must equal i*2^n. Initial resolution i is [{}]. n must be a natural number.".format(resolution, init_resolution)

        cnn.add(Conv2D(channels, (2, 2), padding='same',
                              activation='tanh', kernel_initializer='glorot_normal', use_bias=False))

        # This is the latent z space
        latent = Input(shape=(latent_size, ))

        fake_image_from_latent = cnn(latent)

        # The input-output interface
        self.generator = Model(inputs=latent, outputs=fake_image_from_latent)

    def _build_common_encoder(self, image, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels

        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()

        cnn.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                       input_shape=(channels, resolution, resolution), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        while cnn.output_shape[-1] > min_latent_res:
            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        features = cnn(image)
        return features

    # latent_size is the innermost latent vector size; min_latent_res is latent resolution (before the dense layer).
    def build_reconstructor(self, latent_size, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(channels, resolution, resolution))
        features = self._build_common_encoder(image, min_latent_res)

        # Reconstructor specific
        latent = Dense(latent_size, activation='linear')(features)
        self.reconstructor = Model(inputs=image, outputs=latent)

    def build_discriminator(self, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(channels, resolution, resolution))
        features = self._build_common_encoder(image, min_latent_res)

        # Discriminator specific
        aux = Dense(
            self.nclasses+1, activation='softmax', name='auxiliary'  # nclasses+1. The last class is: FAKE
        )(features)
        self.discriminator = Model(inputs=image, outputs=aux)

    def generate_from_latent(self, latent):
        res = self.generator(latent)
        return res

    def generate(self, c, bg=None):  # c is a vector of classes
        latent = self.generate_latent(c, bg)
        res = self.generator.predict(latent)
        return res

    def generate_latent(self, c, bg=None, n_mix=10):  # c is a vector of classes
        res = np.array([
            np.random.multivariate_normal(self.means[e], self.covariances[e])
            for e in c
        ])

        return res

    def discriminate(self, image):
        return self.discriminator(image)

    def __init__(self, classes, target_class_id,
                 # Set dratio_mode, and gratio_mode to 'rebalance' to bias the sampling toward the minority class
                 # No relevant difference noted
                 dratio_mode="uniform", gratio_mode="uniform",
                 adam_lr=0.00005, latent_size=100,
                 res_dir = "./res-tmp", image_shape=[3,32,32], min_latent_res=8):
        self.gratio_mode = gratio_mode
        self.dratio_mode = dratio_mode
        self.classes = classes
        self.target_class_id = target_class_id  # target_class_id is used only during saving, not to overwrite other class results.
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[0]
        self.resolution = image_shape[1]
        if self.resolution != image_shape[2]:
            print("Error: only squared images currently supported by balancingGAN")
            exit(1)

        self.min_latent_res = min_latent_res

        # Initialize learning variables
        self.adam_lr = adam_lr 
        self.adam_beta_1 = 0.5

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build generator
        self.build_generator(latent_size, init_resolution=min_latent_res)
        self.generator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )

        latent_gen = Input(shape=(latent_size, ))

        # Build discriminator
        self.build_discriminator(min_latent_res=min_latent_res)
        self.discriminator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )

        # Build reconstructor
        self.build_reconstructor(latent_size, min_latent_res=min_latent_res)
        self.reconstructor.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

        # Define combined for training generator.
        fake = self.generator(latent_gen)

        self.discriminator.trainable = False
        self.reconstructor.trainable = False
        self.generator.trainable = True
        aux = self.discriminate(fake)

        self.combined = Model(inputs=latent_gen, outputs=aux)

        self.combined.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )

        # Define initializer for autoencoder
        self.discriminator.trainable = False
        self.generator.trainable = True
        self.reconstructor.trainable = True

        img_for_reconstructor = Input(shape=(self.channels, self.resolution, self.resolution,))
        img_reconstruct = self.generator(self.reconstructor(img_for_reconstructor))

        self.autoenc_0 = Model(inputs=img_for_reconstructor, outputs=img_reconstruct)
        self.autoenc_0.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='mean_squared_error'
        )

    def _biased_sample_labels(self, samples, target_distribution="uniform"):
        distribution = self.class_uratio
        if target_distribution == "d":
            distribution = self.class_dratio
        elif target_distribution == "g":
            distribution = self.class_gratio
            
        sampled_labels = np.full(samples,0)
        sampled_labels_p = np.random.uniform(0, 1, samples)
        for c in list(range(self.nclasses)):
            mask = np.logical_and((sampled_labels_p > 0), (sampled_labels_p <= distribution[c]))
            sampled_labels[mask] = self.classes[c]
            sampled_labels_p = sampled_labels_p - distribution[c]

        return sampled_labels

    def _train_one_epoch(self, bg_train):
        epoch_disc_loss = []
        epoch_gen_loss = []

        for image_batch, label_batch in bg_train.next_batch():

            crt_batch_size = label_batch.shape[0]

            ################## Train Discriminator ##################
            fake_size = int(np.ceil(crt_batch_size * 1.0/self.nclasses))
    
            # sample some labels from p_c, then latent and images
            sampled_labels = self._biased_sample_labels(fake_size, "d")
            latent_gen = self.generate_latent(sampled_labels, bg_train)

            generated_images = self.generator.predict(latent_gen, verbose=0)

            X = np.concatenate((image_batch, generated_images))
            aux_y = np.concatenate((label_batch, np.full(len(sampled_labels) , self.nclasses )), axis=0)

            epoch_disc_loss.append(self.discriminator.train_on_batch(X, aux_y))

            ################## Train Generator ##################
            sampled_labels = self._biased_sample_labels(fake_size + crt_batch_size, "g")
            latent_gen = self.generate_latent(sampled_labels, bg_train)

            epoch_gen_loss.append(self.combined.train_on_batch(
                latent_gen, sampled_labels))

        # return statistics: generator loss,
        return (
            np.mean(np.array(epoch_disc_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0)
        )

    def _set_class_ratios(self):
        self.class_dratio = np.full(self.nclasses, 0.0)
        # Set uniform
        target = 1/self.nclasses
        self.class_uratio = np.full(self.nclasses, target)
        
        # Set gratio
        self.class_gratio = np.full(self.nclasses, 0.0)
        for c in range(self.nclasses):
            if self.gratio_mode == "uniform":
                self.class_gratio[c] = target
            elif self.gratio_mode == "rebalance":
                self.class_gratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown gmode " + self.gratio_mode)
                exit()
                
        # Set dratio
        self.class_dratio = np.full(self.nclasses, 0.0)
        for c in range(self.nclasses):
            if self.dratio_mode == "uniform":
                self.class_dratio[c] = target
            elif self.dratio_mode == "rebalance":
                self.class_dratio[c] = 2 * target - self.class_aratio[c]
            else:
                print("Error while training bgan, unknown dmode " + self.dratio_mode)
                exit()

        # if very unbalanced, the gratio might be negative for some classes.
        # In this case, we adjust..
        if self.gratio_mode == "rebalance":
            self.class_gratio[self.class_gratio < 0] = 0
            self.class_gratio = self.class_gratio / sum(self.class_gratio)
            
        # if very unbalanced, the dratio might be negative for some classes.
        # In this case, we adjust..
        if self.dratio_mode == "rebalance":
            self.class_dratio[self.class_dratio < 0] = 0
            self.class_dratio = self.class_dratio / sum(self.class_dratio)

    def init_autoenc(self, bg_train, gen_fname=None, rec_fname=None):
        if gen_fname is None:
            generator_fname = "{}/{}_decoder.h5".format(self.res_dir, self.target_class_id)
        else:
            generator_fname = gen_fname
        if rec_fname is None:
            reconstructor_fname = "{}/{}_encoder.h5".format(self.res_dir, self.target_class_id)
        else:
            reconstructor_fname = rec_fname

        multivariate_prelearnt = False

        # Preload the autoencoders
        if os.path.exists(generator_fname) and os.path.exists(reconstructor_fname):
            print("BAGAN: loading autoencoder: ", generator_fname, reconstructor_fname)
            self.generator.load_weights(generator_fname)
            self.reconstructor.load_weights(reconstructor_fname)

            # load the learned distribution
            if os.path.exists("{}/{}_means.npy".format(self.res_dir, self.target_class_id)) \
                    and os.path.exists("{}/{}_covariances.npy".format(self.res_dir, self.target_class_id)):
                multivariate_prelearnt = True

                cfname = "{}/{}_covariances.npy".format(self.res_dir, self.target_class_id)
                mfname = "{}/{}_means.npy".format(self.res_dir, self.target_class_id)
                print("BAGAN: loading multivariate: ", cfname, mfname)
                self.covariances = np.load(cfname)
                self.means = np.load(mfname)

        else:
            print("BAGAN: training autoencoder")
            autoenc_train_loss = []
            for e in range(self.autoenc_epochs):
                autoenc_train_loss_crt = []
                for image_batch, label_batch in bg_train.next_batch():

                    autoenc_train_loss_crt.append(self.autoenc_0.train_on_batch(image_batch, image_batch))
                autoenc_train_loss.append(np.mean(np.array(autoenc_train_loss_crt), axis=0))

            autoenc_loss_fname = "{}/{}_autoencoder.csv".format(self.res_dir, self.target_class_id)
            with open(autoenc_loss_fname, 'w') as csvfile:
                for item in autoenc_train_loss:
                    csvfile.write("%s\n" % item)

            self.generator.save(generator_fname)
            self.reconstructor.save(reconstructor_fname)

        layers_r = self.reconstructor.layers
        layers_d = self.discriminator.layers

        for l in range(1, len(layers_r)-1):
            layers_d[l].set_weights( layers_r[l].get_weights() )

        # Organize multivariate distribution
        if not multivariate_prelearnt:
            print("BAGAN: computing multivariate")
            self.covariances = []
            self.means = []

            for c in range(self.nclasses):
                imgs = bg_train.dataset_x[bg_train.per_class_ids[c]]

                latent = self.reconstructor.predict(imgs)

                self.covariances.append(np.cov(np.transpose(latent)))
                self.means.append(np.mean(latent, axis=0))

            self.covariances = np.array(self.covariances)
            self.means = np.array(self.means)

            # save the learned distribution
            cfname = "{}/{}_covariances.npy".format(self.res_dir, self.target_class_id)
            mfname = "{}/{}_means.npy".format(self.res_dir, self.target_class_id)
            print("BAGAN: saving multivariate: ", cfname, mfname)
            np.save(cfname, self.covariances)
            np.save(mfname, self.means)
            print("BAGAN: saved multivariate")

    def _get_lst_bck_name(self, element):
        # Find last bck name
        files = [
            f for f in os.listdir(self.res_dir)
            if re.match(r'bck_c_{}'.format(self.target_class_id) + "_" + element, f)
        ]
        if len(files) > 0:
            fname = files[0]
            e_str = os.path.splitext(fname)[0].split("_")[-1]

            epoch = int(e_str)

            return epoch, fname

        else:
            return 0, None

    def init_gan(self):
        # Find last bck name
        epoch, generator_fname = self._get_lst_bck_name("generator")

        new_e, discriminator_fname = self._get_lst_bck_name("discriminator")

        if new_e != epoch:  # Reload error, restart from scratch
            return 0

        # Load last bck
        try:
            self.generator.load_weights(os.path.join(self.res_dir, generator_fname))
            self.discriminator.load_weights(os.path.join(self.res_dir, discriminator_fname))
            return epoch

        # Return epoch
        except:  # Reload error, restart from scratch (the first time we train we pass from here)
            return 0

    def backup_point(self, epoch):
        # Remove last bck
        _, old_bck_g = self._get_lst_bck_name("generator")
        _, old_bck_d = self._get_lst_bck_name("discriminator")
        try:
            os.remove(os.path.join(self.res_dir, old_bck_g))
            os.remove(os.path.join(self.res_dir, old_bck_d))
        except:
            pass

        # Bck
        generator_fname = "{}/bck_c_{}_generator_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)
        discriminator_fname = "{}/bck_c_{}_discriminator_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)

        self.generator.save(generator_fname)
        self.discriminator.save(discriminator_fname)

    def train(self, bg_train, bg_test, epochs=50):
        if not self.trained:
            self.autoenc_epochs = epochs

            # Class actual ratio
            self.class_aratio = bg_train.get_class_probability()

            # Class balancing ratio
            self._set_class_ratios()
            print("uratio set to: {}".format(self.class_uratio))
            print("dratio set to: {}".format(self.class_dratio))
            print("gratio set to: {}".format(self.class_gratio))

            # Initialization
            print("BAGAN init_autoenc")
            self.init_autoenc(bg_train)
            print("BAGAN autoenc initialized, init gan")
            start_e = self.init_gan()
            print("BAGAN gan initialized, start_e: ", start_e)

            crt_c = 0
            act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
            img_samples = np.array([
                [
                    act_img_samples,
                    self.generator.predict(
                        self.reconstructor.predict(
                            act_img_samples
                        )
                    ),
                    self.generate_samples(crt_c, 10, bg_train)
                ]
            ])
            for crt_c in range(1, self.nclasses):
                act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                new_samples = np.array([
                    [
                        act_img_samples,
                        self.generator.predict(
                            self.reconstructor.predict(
                                act_img_samples
                            )
                        ),
                        self.generate_samples(crt_c, 10, bg_train)
                    ]
                ])
                img_samples = np.concatenate((img_samples, new_samples), axis=0)

            shape = img_samples.shape
            img_samples = img_samples.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))

            save_image_array(
                img_samples,
                '{}/cmp_class_{}_init.png'.format(self.res_dir, self.target_class_id)
            )

            # Train
            for e in range(start_e, epochs):
                print('Epoch {} of {}'
                      .format(self.dratio_mode, self.gratio_mode, e + 1, epochs))
                # train_disc_loss, train_gen_loss = self._train_one_epoch(copy.deepcopy(bg_train))
                train_disc_loss, train_gen_loss = self._train_one_epoch(bg_train)

                # Test: # generate a new batch of noise
                nb_test = bg_test.get_num_samples()
                fake_size = int(np.ceil(nb_test * 1.0/self.nclasses))
                sampled_labels = self._biased_sample_labels(nb_test, "d")
                latent_gen = self.generate_latent(sampled_labels, bg_test)
            
                # sample some labels from p_c and generate images from them
                generated_images = self.generator.predict(
                    latent_gen, verbose=False)
            
                X = np.concatenate( (bg_test.dataset_x, generated_images) )
                aux_y = np.concatenate((bg_test.dataset_y, np.full(len(sampled_labels), self.nclasses )), axis=0)
            
                # see if the discriminator can figure itself out...
                test_disc_loss = self.discriminator.evaluate(
                    X, aux_y, verbose=False)
            
                # make new latent
                sampled_labels = self._biased_sample_labels(fake_size + nb_test, "g")
                latent_gen = self.generate_latent(sampled_labels, bg_test)

                test_gen_loss = self.combined.evaluate(
                    latent_gen,
                    sampled_labels, verbose=False)

                # generate an epoch report on performance
                self.train_history['disc_loss'].append(train_disc_loss)
                self.train_history['gen_loss'].append(train_gen_loss)
                self.test_history['disc_loss'].append(test_disc_loss)
                self.test_history['gen_loss'].append(test_gen_loss)
                print("train_disc_loss {},\ttrain_gen_loss {},\ttest_disc_loss {},\ttest_gen_loss {}".format(
                    train_disc_loss, train_gen_loss, test_disc_loss, test_gen_loss
                ))

                # Save sample images
                if e % 10 == 9:
                    img_samples = np.array([
                        self.generate_samples(c, 10, bg_train)
                        for c in range(0,self.nclasses)
                    ])

                    save_image_array(
                        img_samples,
                        '{}/plot_class_{}_epoch_{}.png'.format(self.res_dir, self.target_class_id, e)
                    )

                # Generate whole evaluation plot (real img, autoencoded img, fake img)
                if e % 10 == 5:
                    self.backup_point(e)
                    crt_c = 0
                    act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                    img_samples = np.array([
                        [
                            act_img_samples,
                            self.generator.predict(
                                self.reconstructor.predict(
                                    act_img_samples
                                )
                            ),
                            self.generate_samples(crt_c, 10, bg_train)
                        ]
                    ])
                    for crt_c in range(1, self.nclasses):
                        act_img_samples = bg_train.get_samples_for_class(crt_c, 10)
                        new_samples = np.array([
                            [
                                act_img_samples,
                                self.generator.predict(
                                    self.reconstructor.predict(
                                        act_img_samples
                                    )
                                ),
                                self.generate_samples(crt_c, 10, bg_train)
                            ]
                        ])
                        img_samples = np.concatenate((img_samples, new_samples), axis=0)

                    shape = img_samples.shape
                    img_samples = img_samples.reshape((-1, shape[-4], shape[-3], shape[-2], shape[-1]))

                    save_image_array(
                        img_samples,
                        '{}/cmp_class_{}_epoch_{}.png'.format(self.res_dir, self.target_class_id, e)
                    )

            self.trained = True

    def generate_samples(self, c, samples, bg = None):
        return self.generate(np.full(samples, c), bg)

    def save_history(self, res_dir, class_id):
        if self.trained:
            filename = "{}/class_{}_score.csv".format(res_dir, class_id)
            generator_fname = "{}/class_{}_generator.h5".format(res_dir, class_id)
            discriminator_fname = "{}/class_{}_discriminator.h5".format(res_dir, class_id)
            reconstructor_fname = "{}/class_{}_reconstructor.h5".format(res_dir, class_id)
            with open(filename, 'w') as csvfile:
                fieldnames = [
                    'train_gen_loss', 'train_disc_loss',
                    'test_gen_loss', 'test_disc_loss'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for e in range(len(self.train_history['gen_loss'])):
                    row = [
                        self.train_history['gen_loss'][e],
                        self.train_history['disc_loss'][e],
                        self.test_history['gen_loss'][e],
                        self.test_history['disc_loss'][e]
                    ]

                    writer.writerow(dict(zip(fieldnames,row)))

            self.generator.save(generator_fname)
            self.discriminator.save(discriminator_fname)
            self.reconstructor.save(reconstructor_fname)

    def load_models(self, fname_generator, fname_discriminator, fname_reconstructor, bg_train=None):
        self.init_autoenc(bg_train, gen_fname=fname_generator, rec_fname=fname_reconstructor)
        self.discriminator.load_weights(fname_discriminator)


