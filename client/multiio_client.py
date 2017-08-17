# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:04:26 2017

@author: x
"""

from grpc.beta import implementations
import numpy as np
import tensorflow as tf

# set host, port or other settings
tf.app.flags.DEFINE_string('server', 'localhost:9000',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

host, port = FLAGS.server.split(':')
print(host,port)
channel = implementations.insecure_channel(host, int(port))


#%% Generate test data
x_fake = np.random.rand(100, 27)

width = 224
height = 224
depth = 3

n_samples = 10

x_data = np.random.rand(n_samples, width,height,depth).astype(np.float32)
x_fake_data = 10 * x_fake[:n_samples,:].astype(np.float32)

input_shape0 = x_data.shape
input_shape1 = x_fake_data.shape

print('Sending request...')
print()
#%% Send request
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'category' # BUILD file ???
request.model_spec.signature_name = 'predict'
request.inputs['input0'].CopyFrom(tf.contrib.util.make_tensor_proto(x_data, 
              shape=input_shape0))
request.inputs['input1'].CopyFrom(tf.contrib.util.make_tensor_proto(x_fake_data, 
              shape=input_shape1))
result = stub.Predict(request, 10.0)  # 10 secs timeout

print(result)