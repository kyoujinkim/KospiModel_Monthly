import tensorflow as tf
#from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, MaxPooling1D, UpSampling1D
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import EarlyStopping
import datetime
import pathlib
import os
import matplotlib.pyplot as pltrm
import pickle
import pandas as pd
import numpy as np
from multiprocessing import Pool

kospi = pd.read_csv('kospi.csv', index_col=0, parse_dates=True, encoding='CP949')

def make_arr(x):
    arr = kospi.iloc[x:x+40].values.flatten()
    arr = arr / arr[20]
    arr = arr + np.random.normal(scale=0.03, size=40)
    arr = arr / arr[20]
    arr = np.reshape(arr, (40,1))

    return arr

class AutoEncoder(Model):
    #https://keras.io/examples/vision/autoencoder/
  def __init__(self):
    super(AutoEncoder, self).__init__()

    input = layers.Input(shape=(40,1))
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(input)
    x = layers.MaxPooling1D(2, padding="same")(x)
    x = layers.Conv1D(16, kernel_size=3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding="same")(x)

    self.encoder = Model(input, encoded, name='Encoder')
    self.encoder.summary()

    x = layers.Conv1DTranspose(1, kernel_size=3, activation='linear', padding='same')(encoded)
    #x = layers.Conv1DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.UpSampling1D(2)(x)
    #decoded = layers.Conv1D(1, kernel_size=3, activation='linear', padding='same')(x)

    self.decoder = Model(encoded, decoded, name='Decoder')
    self.decoder.summary()

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

if __name__ == '__main__':

    data_idx = np.random.randint(low=0, high=len(kospi)-40, size=10000)

    p = Pool(os.cpu_count()-1)
    train = p.map(make_arr, data_idx[:-2000])
    test = p.map(make_arr, data_idx[-2000:])

    train_np = np.array(train)
    test_np = np.array(test)
    #train_tf = tf.data.Dataset.from_tensor_slices(train, train)
    #test_tf = tf.data.Dataset.from_tensor_slices(test, test)

    autoencoder = AutoEncoder()
    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mae')

    history = autoencoder.fit(train_np, train_np,
                              epochs=100000,
                              batch_size=1024,
                              shuffle=True,
                              validation_data=(test_np, test_np),
                              callbacks=[early_stopping])

    autoencoder.save_weights('./model/model')

    with open('./history/history', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)