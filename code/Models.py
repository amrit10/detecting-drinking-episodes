import numpy as np
import tensorflow as tf
from time import time
import math

# Baseline Neural Network 
# 3 hidden layers - 32, 32, 16 units
def cnn_model(train_x, train_y, test_x, test_y, epochs, batch_size):
    # 2 Conv1D layers, dropout layer, max pooling layer, fully connected layer 
    conv_layer1 = tf.keras.layers.Conv1D(filters = 64, kernel_size = 3)
    conv_layer2 = tf.keras.layers.Conv1D(filters = 64, kernel_size = 3)
    dropout = tf.keras.layers.Dropout(0.5)
    max_pooling = tf.keras.layers.MaxPool1D(pool_size=2)
    # fc - fully connected layer
    fc_layer = tf.keras.layers.Dense(units=128, activation = 'leaky_relu')
    base_model = tf.keras.Sequential([conv_layer1, conv_layer2, dropout, max_pooling, fc_layer])
    base_model.compile(loss=tf.keras.losses.BinaryCrossEntropy, optimizer=tf.keras.optimizers.Adam, metrics=tf.keras.metrics.Accuracy)
    base_model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, verbose=1)
    loss, accuracy = base_model.evaluate(test_x, test_y, batch_size = batch_size, verbose=0)
    return loss, accuracy 

def cnn_lstm_model(train_x, train_y, test_x, test_y, epochs, batch_size):
    # Base CNN passed in TimeDistributed layer as input to LSTM 
    # LSTM - 128 units, dropout (0.5), fully connected layer (128 units)
    conv_layer1 = tf.keras.layers.Conv1D(filters = 64, kernel_size = 3)
    conv_layer2 = tf.keras.layers.Conv1D(filters = 64, kernel_size = 3)
    cnn_dropout = tf.keras.layers.Dropout(0.5)
    max_pooling = tf.keras.layers.MaxPool1D(pool_size=2)
    # fc - fully connected layer
    fc_layer = tf.keras.layers.Dense(units=128, activation = 'leaky_relu')
    cnn_model = tf.keras.Sequential([conv_layer1, conv_layer2, cnn_dropout, max_pooling, fc_layer])

    lstm_layer = tf.keras.layers.LSTM(units=128)
    lstm_dropout = tf.keras.layers.Dropout(0.5)
    fc_layer2 = tf.keras.layers.Dense(units=128, activation = 'leaky_relu')
    cnn_lstm_model = tf.keras.Sequential([tf.keras.layers.TimeDistributed(cnn_model), lstm_layer, lstm_dropout, fc_layer2])
    cnn_lstm_model.compile(loss=tf.keras.losses.BinaryCrossEntropy, optimizer=tf.keras.optimizers.Adam, metrics = tf.keras.metrics,Accuracy)
    cnn_lstm_model.fit(train_x, train_y, epochs = epochs, batch_size = batch_size, verbose=1)
    loss, accuracy = cnn_lstm_model.evaluate(test_x, test_y, batch_size = batch_size, verbose=0)
    return loss, accuracy 







