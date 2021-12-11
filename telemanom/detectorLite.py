import numpy as np
import os
import time
import tensorflow as tf
from telemanom import helpers
from telemanom.utility import create_path
from telemanom.channel import Channel
from telemanom.errors import Errors
from telemanom.modeling import Model, LiteModel


class DetectorLite:
    def __init__(self, labels_path=None, result_path='results/', config_path='config.yaml'):
        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.channel = None
        self.mode = 'convert'
        self.labels_path = labels_path
        self.results = []

        #custom configuration values
        self.channel_name = 'A-2'
        self.tfModel_path = create_path(self.config, 'models')
        self.tfLiteModel_path = create_path(self.config, 'models', lib='TFLite')
        print(self.channel_name)
        print(self.config.model_architecture, self.config.n_layers)



    def run(self):

        file = np.load(os.path.join("data", "test", "{}.npy".format(self.channel_name)))
        print('file {}'.format(file.shape))

        #create channel and load dataset
        self.channel = Channel(self.config, self.channel_name)
        self.channel.load_data()
        self.channel.y_hat = np.load(create_path(self.config, 'y_hat')+self.channel_name+'.npy')
        print('y_hat {}'.format(self.channel.y_hat.shape))

        errors = Errors(self.channel, self.config, None)
        errors.process_batches(self.channel)

        result_row = {
            #'run_id': self.id,
            'chan_id': self.channel_name,
            'num_train_values': len(self.channel.X_train) + len(self.channel.X_valid),
            'num_test_values': len(self.channel.X_test),
            'n_predicted_anoms': len(errors.E_seq),
            'normalized_pred_error': errors.normalized,
            'anom_scores': errors.anom_scores,
            'anomaly_sequences': errors.E_seq
        }
        #if self.labels_path:

        print(result_row)


        """
        
            if self.mode == 'compare':
                y_TF = self.channel.y_test
                y_TFLite = np.load(create_path(self.config, 'y_hat', lib='TFLite') + self.channel_name + '.npy')
    
                print(y_TF.shape, y_TFLite.shape)
                max = 0
                sum = 0
                i = 0
    
                for a, b in zip(y_TF, y_TFLite):
                    val = abs(a - b)
                    sum += val
                    i += 1
                    if val > max:
                        max = val
    
                print('Max: {}'.format(max))
                print('Avg: {}'.format(sum / i))
    
            #create TFLite Model
            tfLite_model = LiteModel(self.config, self.channel)
    
            if self.mode.startswith('convert'):
                #load TF model
                tf_model = Model(self.config, self.channel_name, self.channel)
    
                tf_model.batch_predict(self.channel)
                print('Prediction completed')
                print('Shapes {} {}'.format(self.channel.y_test.shape, self.channel.y_hat.shape))
                for a,b in zip(self.channel.y_test, self.channel.y_hat):
                    val = abs(a - b)
                    print(val)
    
    
                #print(tf_model.model.summary())
                #tf_model.load_predictions()
    
                #convert model to TF Lite
                #tfLite_model.convert(tf_model.model)
    
                #print('Model converted')
    
            if self.mode.endswith('predict'):
                # predict using TFLite Model
                tfLite_model.batch_predict(self.channel)
                print(tfLite_model.y_hat.shape)
            
        """

