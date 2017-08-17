# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:35:45 2017

@author: xjz19
"""
#%% 1. Load model weight from h5 file
import keras
keras.backend.set_learning_phase(0) # very important to do this as a first thing
from keras.models import load_model
model = load_model('concat_all.h5')
#model_cf = load_model('model.h5')
#%% 2. Initialize Tensorflow saved_model_builder
import os
import tensorflow as tf
sess = keras.backend.get_session() ## if TF model: sess = tf.Session()
export_path_base = './'
export_model_version = '1'
export_path = os.path.join(export_path_base, export_model_version)
# remove the existing directory for exporting needs a non-existed directory
if os.path.exists(export_path):
    __import__('shutil').rmtree(export_path)
print('Exporting trained model to', export_path)
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
#%% 3. Configure signature
signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input0': tf.saved_model.utils.build_tensor_info(model.input[0]),
                'input1': tf.saved_model.utils.build_tensor_info(model.input[1])},
        outputs={'output0': tf.saved_model.utils.build_tensor_info(model.output[0]),
                 'output1': tf.saved_model.utils.build_tensor_info(model.output[1]),
                 'output2': tf.saved_model.utils.build_tensor_info(model.output[2]),},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
#%% 4. add session, tag_constants and signature
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
           'predict':
               signature})
#%% 5. save builder
builder.save()
