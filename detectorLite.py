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


    def run(self):

        #create channel and load dataset
        self.channel = Channel(self.config, self.channel_name)
        self.channel.load_data()

        #TODO - if self.config.execution == 'convert' or self.config.execution == 'convert_and_predict':
        #load model
        tf_model = Model(self.config, self.channel_name, self.channel)

        #convert model
        tfLite_model = LiteModel(self.config, self.channel)
        tfLite_model.convert(tf_model.model)
        print('From {}Kb to {}Kb'.format(tf_model.size, tfLite_model.size))

        #TODO - if self.config.execution == 'predict' or self.config.execution == 'convert_and_predict':

