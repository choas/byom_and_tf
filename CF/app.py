import os
import json
import requests
import tensorflow as tf
from flask import Flask, globals
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

APP = Flask(__name__)

MODEL_NAME = str(os.getenv('MODEL_NAME', 'xor'))
MODEL_VERSION = str(os.getenv('MODEL_NAME', '1'))
DEPLOYMENT_URL = str(os.getenv('DEPLOYMENT_URL', ''))

def metadata_transformer(metadata):
    additions = []
    token = globals.request.headers.get("Authorization")
    additions.append(('authorization', token))
    return tuple(metadata) + tuple(additions)

@APP.route('/', methods=['GET'])
def hello():
    return "BYOM (" + MODEL_NAME + " version " + MODEL_VERSION + ")"

@APP.route('/', methods=['POST'])
def main():
    deployment_url = DEPLOYMENT_URL
    deployment_url += "/api/v2/modelServers"

    token = globals.request.headers.get("Authorization")
    headers = {
        'Authorization': token,
        'Cache-Control': "no-cache"
    }
    querystring = {"modelName": MODEL_NAME, "modelVersion": MODEL_VERSION}
    response = requests.request("GET", deployment_url,
                                headers=headers,
                                params=querystring)
    model_info = json.loads(response.text)

    model_info = model_info["modelServers"][0]
    endpoint = model_info["endpoints"][0]
    credentials = implementations.ssl_channel_credentials(
        root_certificates=str(endpoint["caCrt"]))
    channel = implementations.secure_channel(
        str(endpoint["host"]),
        int(endpoint["port"]),
        credentials)

    stub = prediction_service_pb2.beta_create_PredictionService_stub(
        channel, metadata_transformer=metadata_transformer)

    data_str = globals.request.get_data().split(",")
    data = []
    for idx in range(len(data_str)):
        data.append(float(data_str[idx]))

    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = 'serving_default'
    tfutil = tf.contrib.util
    t_proto = tfutil.make_tensor_proto(data,
                                       shape=[1, len(data)],
                                       dtype="float")

    request.inputs["input_image"].CopyFrom(t_proto)

    predict = stub.Predict(request, 1500)
    print predict
    res = predict.outputs['dense_2/Sigmoid:0'].float_val[0]
    print res
    return str(res)

PORT = os.getenv('PORT', 5000)
if __name__ == '__main__':
    APP.debug = not os.getenv('PORT')
    APP.run(host='0.0.0.0', port=int(PORT))
