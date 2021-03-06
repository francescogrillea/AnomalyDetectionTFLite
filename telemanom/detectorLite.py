import numpy as np
import pandas as pd
import os
import sys
from time import strftime
from time import gmtime
from concurrent.futures import ThreadPoolExecutor
import statistics



from telemanom import helpers
from telemanom.monitoring import MonitorResources
from telemanom.utility import create_path
from telemanom.channel import Channel
from telemanom.errors import Errors
from telemanom.modeling import Model, LiteModel

import tensorflow as tf

#TODO- add logging



def secondsToStr(t):
    return strftime("%H:%M:%S", gmtime(t))

class DetectorLite:
    def __init__(self, labels_path=None, results_path='results/', config_path='config.yaml'):

        """
        Top-level class for running anomaly detection over a group of channels
        with values stored in .npy files. Also evaluates performance against a
        set of labels if provided.

        Args:
            labels_path (str): path to .csv containing labeled anomaly ranges
                for group of channels to be processed
            result_path (str): directory indicating where to stick result .csv
            config_path (str): path to config.yaml

        Attributes:
            labels_path (str): see Args
            TF_results (list of dicts): holds dicts of results for each channel using TensorFlow model
            TFLite_results (list of dicts): holds dicts of results for each channel using TensorFlow Lite model
            chan_df (dataframe): holds all channel information from labels .csv
            stats (list of dicts): holds dicts of statistics during the execution
            mode (dict): specify the execution mode [predict using TensorFlow model, convert model from TF to TFLite, predict using TFLite model]
            result_tracker (dict): if labels provided, holds results throughout processing for logging
            config (obj):  Channel class object containing train/test data for X,y for a single channel
            y_hat (arr): predicted channel values
            result_path (str): see Args
        """

        print('Python v{}'.format(sys.version))
        print('TensorFlow v{}'.format(tf.__version__))
        print('GPU Support: {}'.format(tf.test.is_built_with_gpu_support()))
        print(tf.config.list_physical_devices())


        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.channel = None
        self.labels_path = labels_path
        self.results_path = results_path
        self.TFLite_results = []
        self.TF_results = []
        self.chan_df = None
        self.stats = {}

        self.mode = {}
        self.mode['test'] = self.config.test
        self.mode['predict_TF'] = self.config.TF_Prediction
        self.mode['convert'] = self.config.Convert
        self.mode['predict_TFLite'] = self.config.TFLite_Prediction


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
        print('{} - {} layer'.format(self.config.model_architecture, self.config.n_layers))


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
        tmp = label_row['anomaly_sequences']

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

        label_row['anomaly_sequences'] = tmp
        return result_row


    def get_results(self, row, path):
        errors = Errors(self.channel, self.config, None, path)
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


    def save_results(self, results_path='results/', mode='w', stats=True, results=True):
        """
            Save stats and prediction
        """

        base_filename = str(self.config.model_architecture) + '_' + str(self.config.n_layers)

        if stats:
            filename = base_filename + '_stats'
            try:
                iteration = len([file for file in os.listdir(results_path) if file.startswith(filename)]) + 1
            except FileNotFoundError:
                print('No file yet')
                iteration = 1
            filename = filename + '_it' + str(iteration)

            stats_df = pd.DataFrame(self.stats)
            stats_df.to_csv('{}.csv'.format(results_path+filename), mode=mode, index=False)

        if results:
            #save TF Lite results
            filename = base_filename + '_TFLite_results'
            tflite_results_df = pd.DataFrame(self.TFLite_results)
            tflite_results_df.to_csv('{}.csv'.format(results_path+filename), mode=mode, index=False)

            #save TF results
            filename = base_filename + '_TF_results'
            tf_results_df = pd.DataFrame(self.TF_results)
            tf_results_df.to_csv('{}.csv'.format(results_path+filename), mode=mode, index=False)



    def run(self):
        hardware = self.config.hardware
        stats = self.init_stats(hardware)

        for i, row in self.chan_df.iterrows():

            channel_name = row.chan_id
            stats['chan_id'] = channel_name

            #create channel and load dataset
            self.channel = Channel(self.config, channel_name)
            self.channel.load_data()

            if self.mode['test']:
                break

            #load TF model
            tf_model = Model(self.config, channel_name, self.channel)
            print(tf_model.model.summary())

            #istantiate TFLite Model
            tfLite_model = LiteModel(self.config, self.channel)

            if self.mode['predict_TF']:
                #== TensorFlow Predictions ==#
                with ThreadPoolExecutor() as executor:
                    #monitor resources while during prediction
                    monitor = MonitorResources(hardware)
                    mem_thread = executor.submit(monitor.measure_usage)

                    try:
                        #prediction on TensorFlow Model
                        print('Predicting with TensorFlow model')
                        tf_model.batch_predict(self.channel)
                        print('Time elapsed: {}'.format(tf_model.prediction_time))
                        path = create_path(self.config, 'smoothed_errors')
                        self.TF_results.append(self.get_results(row,path))
                    finally:
                        monitor.keep_monitoring = False
                        monitor.calculate_avg()

                        stats['TF Prediction Time'].append(tf_model.prediction_time)
                        if self.config.hardware == 'cpu':
                            stats['avg CPU% during TF prediction'].append(monitor.cpu)
                        if self.config.hardware == 'gpu':
                            stats['avg GPU% during TF prediction'].append(monitor.gpu)
                        stats['avg RAM used during TF prediction'].append(monitor.ram)

            if self.mode['convert']:
                #== Convert to TensorFlow Lite ==#
                with ThreadPoolExecutor() as executor:
                    # monitor resources while during converion
                    monitor = MonitorResources(hardware)
                    mem_thread = executor.submit(monitor.measure_usage)
                    try:
                        # convert model to TF Lite
                        print('Convert to TensorFlow Lite')
                        tfLite_model.convert(tf_model.model)
                        print('Time elapsed: {}'.format(tfLite_model.conversion_time))
                        print('Conversion completed (from {}Kb to {}Kb)'.format(tf_model.size, tfLite_model.size))
                    finally:
                        monitor.keep_monitoring = False
                        monitor.calculate_avg()

                        stats['conversion Time'].append(tfLite_model.conversion_time)
                        stats['avg CPU% during conversion'].append(monitor.cpu)
                        stats['avg RAM used during conversion'].append(monitor.ram)
                        stats['TF size'].append(tf_model.size)
                        stats['TFLite size'].append(tfLite_model.size)

            if self.mode['predict_TFLite']:
                #== TensorFlow Lite Predictions ==#
                with ThreadPoolExecutor() as executor:
                    # monitor resources while during prediction
                    monitor = MonitorResources(hardware)
                    mem_thread = executor.submit(monitor.measure_usage)
                    try:
                        # predict using TFLite Model
                        print('Predicting with TensorFlowLite model')
                        tfLite_model.batch_predict(self.channel)
                        print('Time elapsed: {}'.format(tfLite_model.prediction_time))
                        path = create_path(self.config, 'smoothed_errors', lib='TFLite')
                        self.TFLite_results.append(self.get_results(row, path))

                    finally:
                        monitor.keep_monitoring = False
                        monitor.calculate_avg()

                        if self.config.hardware == 'cpu':
                            stats['avg CPU% during TFLite prediction'].append(monitor.cpu)
                        if self.config.hardware == 'gpu':
                            stats['avg GPU% during TFLite prediction'].append(monitor.gpu)

                        stats['TFLite prediction Time'].append(tfLite_model.prediction_time)
                        stats['avg RAM used during TFLite prediction'].append(monitor.ram)


            self.stats = stats
            #self.calculate_last_row()
            break

        self.save_results()




    def calculate_last_row(self):
        """
        Useful if must iterate over the same channel
        :return: calculate avg and stdev of the iterations and put it in the last row of the results file
        """

        for key in self.stats:
            if key == 'chan_id':
                continue
            #print('{}--> {}'.format(key,self.stats[key]))
            try:
                avg = statistics.mean(self.stats[key])
                stdev = statistics.stdev(self.stats[key])
            except(ZeroDivisionError):
                avg = 0
                stdev = 0
            except(statistics.StatisticsError):
                print('Not enough values to calculate avg or stdev for {}. Setting to 0'.format(key))
                avg = 0
                stdev = 0


            if key.endswith('Time'):
                avg = secondsToStr(avg)
                stdev = secondsToStr(stdev)
                for i in range(len(self.stats[key])):
                    self.stats[key][i] = secondsToStr(self.stats[key][i])

            self.stats[key].append([avg, stdev])

    def init_stats(self, hardware):
        stats = {
            'chan_id': None,
        }

        if self.mode['predict_TF']:
            stats['TF Prediction Time'] = []
            stats['avg RAM used during TF prediction'] = []
            if hardware == 'cpu':
                stats['avg CPU% during TF prediction'] = []
            elif hardware == 'gpu':
                stats['avg GPU% during TF prediction'] = []


        if self.mode['convert']:
            stats['conversion Time'] = []
            stats['avg CPU% during conversion'] = []
            stats['avg RAM used during conversion'] = []
            stats['TF size'] = []
            stats['TFLite size'] = []

        if self.mode['predict_TFLite']:
            stats['TFLite prediction Time'] = []
            stats['avg RAM used during TFLite prediction'] = []
            if hardware == 'cpu':
                stats['avg CPU% during TFLite prediction'] = []
            elif hardware == 'gpu':
                stats['avg GPU% during TFLite prediction'] = []

        return stats