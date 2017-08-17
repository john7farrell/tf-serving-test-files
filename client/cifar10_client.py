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

import keras
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10

print('x_train shape:', x_train.shape)
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

width = 32
height = 32
depth = 3


n_samples = 10
x_data = x_test[:n_samples,:,:,:]
y_data = y_test[:n_samples,:]

input_shape = x_data.shape

print('Sending request...')
print()
#%% Send request
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'cifar10' # BUILD file ???
request.model_spec.signature_name = 'predict'
request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(x_data, 
              shape=input_shape))
result = stub.Predict(request, 10.0)  # 10 secs timeout

print(result)