from flask import Flask, request
from flask_restful import Resource, Api
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import load_model
import json

app = Flask(__name__)
api = Api(app)

# prediction api call


class prediction(Resource):
    def __init__(self) -> None:
        request_data = request.get_json()
        device_id = request_data['device_id']
        self.model = load_model('../model/model' + device_id + ".h5")
        # TODO: Read the scaler from scaler.py
        self.scaler = MinMaxScaler()

    def _predict(self, data, count, forecast=[]):
        forecasted_values = self.model.predict(data)
        val = forecasted_values.reshape(-1, 1)[0:1]
        forecast.append(val[0])
        if count != 0:
            return self._predict(
                data=np.concatenate(
                    [forecast[::-1], data]),
                count=count - 1,
                forecast=forecast
            )
        return forecast

    def _build_response(self, forecasted_values):
        forecast = []
        for f in forecasted_values:
            forecast.append(self.scaler.inverse_transform([f]).tolist())

        return json.dumps({
            'prediction': forecast
        })

    def post(self):
        request_data = request.get_json()

        readings = request_data['readings']
        data = np.array(readings)
        input_data = data.reshape(-1, 1)
        count = request_data['count']
        if not (np.min(input_data) >= 0 and np.max(input_data) <= 1):
            # if the data has not been transformed, use MinMaxScaler to transform the data
            data_transformed = self.scaler.fit_transform(input_data)
        else:
            # if the data has already been transformed, use the original data
            data_transformed = input_data

        forecasted_values = self._predict(data_transformed, count, [])
        return self._build_response(forecasted_values)


@app.route('/flask-health-check')
def flask_health_check():
    return "success"


api.add_resource(prediction, '/predict')

if __name__ == '__main__':
    app.run()
