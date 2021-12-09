import tensorflow as tf
from telemanom import helpers
from telemanom.channel import Channel
from telemanom.modeling import Model
import numpy as np
import os
import time


class DetectorLite:
    def __init__(self, labels_path=None, result_path='results/', config_path='config.yaml'):
        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.tf_predictions = None
        self.tfLite_predictions = None
        self.tfModel_size = 0
        self.tfLiteModel_size = 0
        self.conversion_time = 0 #seconds

        #custom configuration values
        self.architecture = 'LSTM_1L'
        self.channel_name = 'A-1'
        self.tfModel_path = self.create_path('models')
        self.tfLiteModel_path = self.create_path('models', lib='TFLite')

    def create_path(self, obj, lib='TF'):
        #lib = TF, TFLite respectly for TensorFlow and TensorFlow Lite folder
        #obj = model, y_hat, smoothed_errors

        folder = self.config.model_architecture+'_'+str(self.config.n_layers)+'L'
        if self.config.model_architecture == 'ESN':
            if self.config.serialization == True:
                folder = folder+'_SER'
        path = 'data/'+lib+'/'+folder+'/'+obj+'/'
        return path

    def convert_model(self, tf_model):
        # convert model from TF to TFLite
        start_time = time.time()
        converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        delta_time = time.time() - start_time
        self.conversion_time = delta_time
        print(self.conversion_time)

        # save TFLite model
        TFLite_MODEL_FILE = self.tfLiteModel_path+self.channel_name+'.tflite'
        with open(TFLite_MODEL_FILE, 'wb') as f:
            f.write(tflite_model)
        self.tfLiteModel_size = int(os.path.getsize(TFLite_MODEL_FILE) / 1024)

        return tflite_model


    def run(self):
        #load F predictions
        self.tf_predictions = np.load(self.create_path('y_hat')+self.channel_name+'.npy')
        print(self.tf_predictions)

        #create channel and load dataset
        channel = Channel(self.config, self.channel_name)
        channel.load_data()

        #TODO - if self.config.execution == 'convert' or self.config.execution == 'convert_and_predict':
        #load model
        model = Model(self.config, self.channel_name, channel)
        tf_model = model.model

        #convert model
        tfLite_model = self.convert_model(tf_model)
        print('From {}Kb to {}Kb'.format(self.tfModel_size, self.tfLiteModel_size))

        #TODO - if self.config.execution == 'predict' or self.config.execution == 'convert_and_predict':

