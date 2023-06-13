from flask import Flask, jsonify,redirect, session,request
import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import os
import tensorflow as tf
from tensorflow.keras.layers import Cropping1D,Embedding, Flatten, Dense, concatenate, Dropout,GlobalAveragePooling1D,GlobalMaxPooling1D,LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.debug = True

@app.route('/train', methods=['GET'])


def train():
    #Import the data set
    df = pd.read_csv('Reviews3.csv')

    # Dropping the columns
    df = df.drop(['Id','HelpfulnessNumerator','HelpfulnessDenominator'], axis = 1) 

    # Data model preparation as per requirement on number of minimum ratings
    counts = df['UserId'].value_counts()
    df_final = df[df['UserId'].isin(counts[counts >= 40].index)]

    # Split the data into training and test sets
    train_data, test_data = train_test_split(df_final, test_size=0.1, random_state=0)

    # Create a dictionary mapping user IDs to unique indices
    user_ids = df_final['UserId'].unique()
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    index_to_user_id = {index: user_id for index, user_id in enumerate(user_ids)}

    # Create a dictionary mapping product IDs to unique indices
    product_ids = df_final['ProductId'].unique()
    product_id_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
    index_to_product_id = {index: product_id for index, product_id in enumerate(product_ids)}

    # Convert user and product IDs to indices and food names in the dataframe
    train_data['user_index'] = train_data['UserId'].map(user_id_to_index)
    train_data['product_index'] = train_data['ProductId'].map(product_id_to_index)

    test_data['user_index'] = test_data['UserId'].map(user_id_to_index)
    test_data['product_index'] = test_data['ProductId'].map(product_id_to_index)

    # Prepare the training data
    x_train_user = train_data['user_index'].values
    x_train_product = train_data['product_index'].values
    x_train = np.concatenate((x_train_user.reshape(-1, 1), x_train_product.reshape(-1, 1)), axis=1)
    y_train = train_data['Score'].values

    scaler = MinMaxScaler()
    y_train = scaler.fit_transform(y_train.reshape(-1, 1))

    # Prepare the test data
    x_test_user = test_data['user_index'].values
    x_test_product = test_data['product_index'].values
    x_test = np.concatenate((x_test_user.reshape(-1, 1), x_test_product.reshape(-1, 1)), axis=1)
    y_test = test_data['Score'].values

    # Set the input shape
    num_users = len(user_ids)
    num_products = len(product_ids)
    embedding_dim = 512

    # Create the embedding layer
    embedding_layer = Embedding(num_users + num_products +1, embedding_dim, input_length=2)

    # Create the model
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='linear'))

    # Setup for tensorboard
    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    #add optimizer for model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    #add delta for optimizer
    delta = 1.0

    # Compile the model
    model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(delta=delta))

    # Train the model
    history = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.1,callbacks=[tensorboard_callback])

    # Convert model to  H5
    export_dir = 'saved_model/1'
    model.save(export_dir+'/recommender_model.h5')

    # Convert result acc
    accuracy = {'loss':history.history['loss']}
    return jsonify(accuracy)

if __name__ == '__main__':
    app.run()

