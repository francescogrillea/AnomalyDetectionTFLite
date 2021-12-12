import numpy as np
import pandas as pd
import os
import time
import tensorflow as tf
from telemanom import helpers
from telemanom.utility import create_path
from telemanom.channel import Channel
from telemanom.errors import Errors
from telemanom.modeling import Model, LiteModel

#TODO def write_csv():


class DetectorLite:
    def __init__(self, labels_path=None, results_path='results/', config_path='config.yaml'):

        #TODO- inserire commenti

        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.channel = None
        self.mode = 'test' #[test, convert, predict, convert_predict]
        self.labels_path = labels_path
        self.results_path = results_path
        self.results = []
        self.save_results = True
        self.chan_df = None


        if self.labels_path:
            self.chan_df = pd.read_csv(labels_path)
        else:
            chan_ids = [x.split('.')[0] for x in os.listdir('data/test/')]
            self.chan_df = pd.DataFrame({"chan_id": chan_ids})


        self.result_tracker = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_negatives': 0
        }


        #custom configuration values
        self.tfModel_path = create_path(self.config, 'models')
        self.tfLiteModel_path = create_path(self.config, 'models', lib='TFLite')
        print(self.config.model_architecture, self.config.n_layers, self.config.hp_research_id)


    def evaluate_sequences(self, errors, label_row):
        """
        Compare identified anomalous sequences with labeled anomalous sequences.

        Args:
            errors (obj): Errors class object containing detected anomaly
                sequences for a channel
            label_row (pandas Series): Contains labels and true anomaly details
                for a channel

        Returns:
            result_row (dict): anomaly detection accuracy and results
        """

        result_row = {
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0,
            'fp_sequences': [],
            'tp_sequences': [],
            'num_true_anoms': 0
        }

        matched_true_seqs = []

        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        result_row['scores'] = errors.anom_scores


        if len(errors.E_seq) == 0:
            result_row['false_negatives'] = result_row['num_true_anoms']

        else:
            true_indices_grouped = [list(range(e[0], e[1]+1)) for e in label_row['anomaly_sequences']]
            true_indices_flat = set([i for group in true_indices_grouped for i in group])

            for e_seq in errors.E_seq:
                i_anom_predicted = set(range(e_seq[0], e_seq[1]+1))

                matched_indices = list(i_anom_predicted & true_indices_flat)
                valid = True if len(matched_indices) > 0 else False

                if valid:

                    result_row['tp_sequences'].append(e_seq)

                    true_seq_index = [i for i in range(len(true_indices_grouped)) if
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]

                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] += 1

                else:
                    result_row['fp_sequences'].append([e_seq[0], e_seq[1]])
                    result_row['false_positives'] += 1

            result_row["false_negatives"] = len(np.delete(label_row['anomaly_sequences'],
                                                          matched_true_seqs, axis=0))


        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row


    def get_results(self, row):
        errors = Errors(self.channel, self.config, None)
        errors.process_batches(self.channel)

        result_row = {
            #'run_id': self.id,
            'chan_id': row.chan_id,
            'num_train_values': len(self.channel.X_train) + len(self.channel.X_valid),
            'num_test_values': len(self.channel.X_test),
            'n_predicted_anoms': len(errors.E_seq),
            'normalized_pred_error': errors.normalized,
            'anom_scores': errors.anom_scores,
        }
        if self.labels_path:
            result_row = {**result_row, **self.evaluate_sequences(errors, row)}
            result_row['spacecraft'] = row['spacecraft']
            result_row['anomaly_sequences'] = row['anomaly_sequences']
            result_row['class'] = row['class']
        else:
            result_row['anomaly_sequences']: errors.E_seq

        return result_row


    def run(self):

        for i, row in self.chan_df.iterrows():
            channel_name = row.chan_id
            print(row.chan_id)

            #create channel and load dataset
            self.channel = Channel(self.config, channel_name)
            self.channel.load_data()
            #self.channel.y_hat = np.load(create_path(self.config, 'y_hat')+channel_name+'.npy')

            # create TFLite Model
            tfLite_model = LiteModel(self.config, self.channel)

            if self.mode.startswith('convert'):
                # load TF model
                tf_model = Model(self.config, channel_name, self.channel)
                print('Loading TensorFlow model: Done')
                #print(tf_model.model.summary())
                #tf_model.batch_predict(self.channel)

                # convert model to TF Lite
                tfLite_model.convert(tf_model.model)
                print('Conversion to TensorFlow Lite: Done (from {}Kb to {}Kb)'.format(tf_model.size, tfLite_model.size))

            if self.mode.endswith('predict'):
                # predict using TFLite Model
                print('Predicting with TensorFlowLite model')
                tfLite_model.batch_predict(self.channel)

            if self.mode == 'test':
                self.channel.y_hat = np.load(create_path(self.config, 'y_hat')+channel_name+'.npy')

            self.results.append(self.get_results(row))
            #TODO da eliminare
            break

        print(self.results)

        if self.save_results:
            with open('results/myFile.txt', 'a') as f:
                if self.mode.endswith('predict'):
                    data = 'TF_lite '
                else:
                    data = 'TF '
                data = data + str(self.config.model_architecture) + str(self.config.n_layers) + ' ' + channel_name + '- '+ str(self.results) + '\n'
                f.write(data)
                f.close()
