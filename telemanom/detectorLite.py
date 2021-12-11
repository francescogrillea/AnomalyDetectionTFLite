import numpy as np
import time
import tensorflow as tf
from telemanom import helpers
from telemanom.utility import create_path
from telemanom.channel import Channel
from telemanom.modeling import Model, LiteModel


class DetectorLite:
    def __init__(self, labels_path=None, result_path='results/', config_path='config.yaml'):
        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.channel = None
        self.mode = 'convert_predict'

        #custom configuration values
        self.channel_name = 'A-1'
        self.tfModel_path = create_path(self.config, 'models')
        self.tfLiteModel_path = create_path(self.config, 'models', lib='TFLite')
        print(self.config.model_architecture, self.config.n_layers)


    def compare_predictions(self):
        y_TF = np.load(create_path(self.config, 'y_hat')+self.channel_name+'.npy')
        y_TFLite = np.load(create_path(self.config, 'y_hat', lib='TFLite')+self.channel_name+'.npy')

        print(y_TF.shape, y_TFLite.shape)
        max, sum, i = (0,0,0)

        for a,b in zip(y_TF, y_TFLite):
            val = abs(a-b)
            sum += val
            i += 1
            if val > max:
                max = val

        print('Max: {}'.format(max))
        print('Avg: {}'.format(sum/i))

    def run(self):

        #create channel and load dataset
        self.channel = Channel(self.config, self.channel_name)
        self.channel.load_data()

        #create TFLite Model
        tfLite_model = LiteModel(self.config, self.channel)

        if self.mode.startswith('convert'):
            #load TF model
            tf_model = Model(self.config, self.channel_name, self.channel)
            #print(tf_model.model.summary())
            #tf_model.load_predictions()

            #convert model to TF Lite
            tfLite_model.convert(tf_model.model)

            print('Model converted')

        if self.mode.endswith('predict'):
            # predict using TFLite Model
            tfLite_model.batch_predict(self.channel)
            print(tfLite_model.y_hat.shape)
