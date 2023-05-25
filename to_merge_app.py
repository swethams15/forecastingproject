import tensorflow
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import h5py
from keras.models import load_model
import pandas as pd
import numpy as np
from flask_cors import CORS
from numpy import float32
from sklearn.preprocessing import MinMaxScaler
import json


app = Flask(__name__)
#
CORS(app)
# creating an API object
api = Api(app)


# prediction api call
class prediction(Resource):
    def post(self):
        request_data = request.get_json()

        readings = request_data['readings']
        data = np.array(readings)
        input_data = data.reshape(-1, 1)
        count = request_data['count']
        if not (np.min(input_data) >= 0 and np.max(input_data) <= 1):
            # if the data has not been transformed, use MinMaxScaler to transform the data
            scaler = MinMaxScaler()
            data_transformed = scaler.fit_transform(input_data)
        else:
            # if the data has already been transformed, use the original data
            data_transformed = input_data

        model = load_model('model.h5')
            # prediction = np.zeros(144)
        prediction = model.predict(data_transformed)
        for f in prediction:
            response = {'prediction': scaler.inverse_transform(f).tolist()}
            return json.dumps(response)

        #data =request.args.get('readings', default=None, type=float)
        #count = request.args.post('count', default=144, type=int)
        #current = request.form['current']

        #if NotNone throw error

        # data = np.array(data)
        # input_data = data.reshape(-1, 1)
        # # for i in range(144):
        # #     input_data = data[-144:]
        #
        # if not (np.min(input_data) >= 0 and np.max(input_data) <= 1):
        #     # if the data has not been transformed, use MinMaxScaler to transform the data
        #     scaler = MinMaxScaler()
        #     data_transformed = scaler.fit_transform(input_data)
        # else:
        #     # if the data has already been transformed, use the original data
        #     data_transformed = input_data
        #
        # # print the transformed data
        # #print(data_transformed)
        # model = load_model('model.h5')
        # #prediction = np.zeros(144)
        # prediction = model.predict(data_transformed)
        # #new_prediction = prediction.reshape((1, 1, 144))
        #     #predictions[i] = prediction
        #     #print(prediction)
        # for f in prediction:
        #     response = {'prediction': scaler.inverse_transform(f).tolist()}
        #     return json.dumps(response)

                #print(np.concatenate([input_data, [prediction]]))


# data api
class getData(Resource):
    def get(self):
            # df = pd.read_excel('data.xlsx')
            # df =  df.rename({'Marketing Budget': 'budget', 'Actual Sales': 'sale'}, axis=1)  # rename columns
            # #print(df.head())
            # #out = {'key':str}
            # res = df.to_json(orient='records')
            # #print( res)
            return 'success'

#
api.add_resource(getData, '/api')
api.add_resource(prediction, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
