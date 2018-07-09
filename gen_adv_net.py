import sys
import os
from ganfinance.build_GAN_models import *

import keras
from keras.models import Sequential
from keras.layers import Dropout, Flatten
from keras.layers.core import Activation, Dense, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import backend as K
from random import randint

import numpy as np
import random
import pandas as pd


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    df.columns = ['X', 'Y']
    return df

def generate_sample(dataset, size=128, batches=32):
    upper_bound = dataset.shape[0] - 1 - size
    sample = []
    for index in iter(randint(0, upper_bound) for _ in range(batches)):
        sample.append(dataset[index: index + size])
    return np.array(sample)

def train(data_file, out_loc, batch_size, epochs, lr_gen, lr_desc, lr_both):
    batch_size = int(batch_size)
    epochs = int(epochs)

    ######### Initalize the models, and then combine them to create discriminator_on_generator
    generator = gen_model_stock(lr=lr_gen)
    discriminator = desc_model_stock(lr=lr_desc)
    discriminator_on_generator = build_gen_desc_stock(generator, discriminator,  lr=lr_both)
    discriminator.trainable = True

    # Save the training loss:
    desc_loss_hist=[]
    gen_loss_hist=[]

    ######### Train the model:
    ## Maybe it would be best to just make two data generators
    ## Really I just want to use the keras function to nicely output training progress
    #data=pd.read_csv(data_file, sep=',').as_matrix().astype(float)[0:80000,1:] # leave some out just incase
    data = pd.read_csv(data_file, index_col='Date')
    data = data['Adj Close']
    mean = data.mean()
    print('data shape: ', data.shape)
    data=np.expand_dims(data, axis=2)
    for epoch in range(epochs):
        print('epoch:', epoch)

        # See how many batches you'll need to cover the whole set
        for index in range(int(data.shape[0]/batch_size)):
            # Generate noise
            sign = 2 * np.random.binomial(1, 0.5, size=(32, 1, 730)) - 1
            noise = (np.random.uniform(0, 1, size=(32, 1, 730)) * sign) + mean
            # Now generate the random images:
            generated_data = generator.predict(noise)
            #print('generated:', generated_data.shape)
            # get real images, and join everything together with the ys for the discriminator
            #true_batch = data[index*batch_size:(index+1)*batch_size]
            #true_batch = np.expand_dims(true_batch, axis=2)
            true_batch = generate_sample(data, size=730)
            #print('true:', true_batch.shape)
            x_data = np.concatenate((true_batch, generated_data)).astype(float)
            #print('x_data:', x_data.shape)
            y_data=np.append(np.repeat(0, batch_size), np.repeat(1, batch_size), axis=0)
            y_data=np.eye(2)[y_data].astype(float)
            d_loss = discriminator.train_on_batch(x_data, y_data) # Is it better to randomize this???
            if index % 500 == 0:
                desc_loss_hist.append(d_loss)
                print("batch %d d_loss : %f" % (index, d_loss))

        ## Now train the generator:
        discriminator.trainable = False
        sign = 2 * np.random.binomial(1, 0.5, size=(32, 1, 730)) - 1
        noise = (np.random.uniform(0, 1, size=(32, 1, 730)) * sign) + mean
        # for i in range(batch_size):
        #     noise[i, :] = np.random.uniform(0, 1, 730)
        y_noise = np.eye(2)[np.repeat(1, batch_size)].astype(float)
        g_loss = discriminator_on_generator.train_on_batch(noise, y_noise)
        discriminator.trainable = True
        if index % 500 == 0:
            gen_loss_hist.append(g_loss)
            print("batch %d g_loss       : %f" % (index, g_loss))

        generator_hist_file = os.path.join(out_loc, 'generator_hist')
        discriminator_hist_file = os.path.join(out_loc, 'discriminator_hist')
        np.save(generator_hist_file, desc_loss_hist)
        np.save(discriminator_hist_file, gen_loss_hist)

        generator_file=os.path.join(out_loc, 'generator')
        discriminator_file=os.path.join(out_loc, 'discriminator')
        generator.save_weights(generator_file, True)
        discriminator.save_weights(discriminator_file, True)


# All done in the shit way with no variable input functions
def generate(out_loc, num, data_file, only_good_ones=True):
    data = pd.read_csv(data_file, index_col='Date')
    data = data['Adj Close']
    mean = data.mean()
    generated_samples = []
    num = int(num)

    generator = gen_model_stock()
    generator_file = os.path.join(out_loc, 'generator')
    print(generator_file)
    generator.load_weights(generator_file)

    if only_good_ones: #take only the top 1 in every 10 generated images
        discriminator = desc_model_stock()
        discriminator_file = os.path.join(out_loc, 'discriminator')
        discriminator.load_weights(discriminator_file)
        for index in range(num):
            sign = 2 * np.random.binomial(1, 0.5, size=(10, 730)) - 1
            noise = (np.random.uniform(0, 1, size=(10, 730)) * sign) + mean
            gen_data = generator.predict(noise, verbose=1)
            gen_data_pred = discriminator.predict(gen_data, verbose=1)[:,0] # hot-one, 1st col is p(true)
            print(gen_data.shape)


            # Now sort the images based on how good they were, and take the best one
            order_list=range(10)
            gen_data_pred, inds = (list(t) for t in zip(*sorted(zip(gen_data_pred, order_list))))
            idx = int(inds[9])
            good_gen_data = gen_data[idx, :, :]
            generated_samples.append(good_gen_data)

            # Whats going on?
            print('Probability of true: ', gen_data_pred[0])


    else:
        for index in range(num):
            sign = 2 * np.random.binomial(1, 0.5, size=(10, 730)) - 1
            noise = (np.random.uniform(0, 1, size=(10, 730)) * sign) + mean
            data = generator.predict(noise, verbose=1)
            generated_samples.append(data)
    generated_samples=np.array(generated_samples)

    generated_samples=np.array(generated_samples)
    print('shape of generated_samples: ', generated_samples.shape)


    out_file = os.path.join(out_loc, 'generated_data')
    np.save(out_file, generated_samples)


def run(data_file, out_loc, num_gen, batch_size, epochs, lr_gen=.01, lr_desc=0.0005, lr_both=0.0005):
    # use 50, 1600- 320steps*1000per batch
    train(data_file, out_loc, batch_size, epochs, lr_gen, lr_desc, lr_both)
    generate(out_loc=out_loc, num=num_gen, only_good_ones=True)


if __name__ == "__main__":
    data_file = sys.argv[1]
    out_loc = sys.argv[2]
    num_gen = sys.argv[3]
    batch_size = sys.argv[4]
    epochs = sys.argv[5]

    run(data_file, out_loc, num_gen, batch_size, epochs)
