from flask import Flask, jsonify, redirect, session,request
import requests
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.debug = True

@app.route('/predict', methods=['GET'])


def predict():
    # Load the model
    model = tf.keras.models.load_model('saved_model/1/recommender_model.h5')

    #Import the data set
    df = pd.read_csv('Reviews3.csv')

    #Dropping the columns
    df = df.drop(['Id','HelpfulnessNumerator','HelpfulnessDenominator'], axis = 1) 
    
    # Set the user ID
    specific_user_id = 'A2A9X58G2GTBLP' #example can be changed

    #preprocess input data
    if specific_user_id in df['userId'] and df[df['userId'] == specific_user_id].shape[0] < 5:
        # Top 10 based on rating
        most_rated = df.groupby('ProductId').size().sort_values(ascending=False)[:10]
        final_result = most_rated.index

    else:
        counts = df['UserId'].value_counts()
        df_final = df[df['UserId'].isin(counts[counts >= 40].index)]

        # Create a dictionary mapping user IDs to unique indices
        user_ids = df_final['UserId'].unique()
        user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        index_to_user_id = {index: user_id for index, user_id in enumerate(user_ids)}

        # Create a dictionary mapping product IDs to unique indices
        product_ids = df_final['ProductId'].unique()
        product_id_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
        index_to_product_id = {index: product_id for index, product_id in enumerate(product_ids)}
        
        # Convert user and product IDs to indices and food names in the dataframe
        df_final['user_index'] = df_final['UserId'].map(user_id_to_index)
        df_final['product_index'] = df_final['ProductId'].map(product_id_to_index)

        uidx= df_final['user_index'].values.astype(np.int64)
        pidx = df_final['product_index'].values.astype(np.int64)

        # Create a new DataFrame with converted data arrays
        df_converted = pd.DataFrame({'UserId': uidx, 'ProductId': pidx, 'Score': 0})

        # Create pivot table with the converted DataFrame
        final_ratings_matrix = pd.pivot_table(df_converted, index='UserId', columns='ProductId', values='Score')
        final_ratings_matrix.fillna(0, inplace=True)

        array3 = final_ratings_matrix.reset_index().melt(id_vars=['UserId'], value_vars=final_ratings_matrix.columns)
        array3 = array3[['UserId', 'ProductId']].values.astype(np.int64)
        
        # Filter the array3 for the specific user ID
        filtered_array3 = array3[array3[:, 0] == user_id_to_index[specific_user_id]]

        # Perform predictions
        predictions = model.predict(filtered_array3)
        
        # Inverse transform the scaled ratings to get the actual ratings
        scaler = MinMaxScaler()
        score = scaler.fit_transform(df['Score'].values.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)

        #make prediction result to df
        df_predicted = pd.DataFrame(filtered_array3, columns=['UserId', 'ProductId'])
        df_predicted['PredictedRatings']=predictions

        # Rename the columns back to 'UserId' and 'ProductId'
        df_predicted = df_predicted.rename(columns={'UserId': 'user_index', 'ProductId': 'product_index'})

        # Convert the user index back to 'UserId'
        df_predicted['user_index'] = df_predicted['user_index'].map(index_to_user_id)
        # Convert the user index back to 'UserId'
        df_predicted['product_index'] = df_predicted['product_index'].map(index_to_product_id)

        df_predicted = df_predicted.rename(columns={'user_index': 'UserId',
                                                'product_index':'ProductId'})
        df_predicted = df_predicted.sort_values(by='PredictedRatings',ascending=False)
        final_result = df_predicted['ProductId'][:10].values

    # Convert predictions to a JSON response
    response = {'predictions': final_result.tolist()}
    return jsonify(response)


if __name__ == '__main__':
    app.run()
