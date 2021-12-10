import numpy as np

from telemanom import helpers
from telemanom.utility import create_path
from telemanom.channel import Channel
from telemanom.modeling import Model, LiteModel


class DetectorLite:
    def __init__(self, labels_path=None, result_path='results/', config_path='config.yaml'):
        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.channel = None

        #custom configuration values
        self.channel_name = 'A-1'
        self.tfModel_path = create_path(self.config, 'models')
        self.tfLiteModel_path = create_path(self.config, 'models', lib='TFLite')
        print(self.config.model_architecture, self.config.n_layers)


    def run(self):

        #create channel and load dataset
        self.channel = Channel(self.config, self.channel_name)
        self.channel.load_data()
        #X_TEST.shape (8380, 250, 25)

        #TODO - if self.config.execution == 'convert' or self.config.execution == 'convert_and_predict':
        #load model
        tf_model = Model(self.config, self.channel_name, self.channel)
            #inputshape = (None, None, 25)
            #outputshape = (None, 10)
        #tf_model.load_predictions()


        #create TFLite Model
        tfLite_model = LiteModel(self.config, self.channel)

        #convert model to TensorFlow Lite
        tfLite_model.convert(tf_model.model)
        print('From {}Kb to {}Kb'.format(tf_model.size, tfLite_model.size))

        #predict using TFLite Model
        tfLite_model.batch_predict(self.channel)
        print(tfLite_model.y_hat.shape)

        #TODO - if self.config.execution == 'predict' or self.config.execution == 'convert_and_predict':
