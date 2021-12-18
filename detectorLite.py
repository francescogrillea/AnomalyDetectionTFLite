import numpy as np
import pandas as pd
import os
from time import strftime
from time import gmtime
from concurrent.futures import ThreadPoolExecutor
from telemanom import helpers
from telemanom.monitoring import MonitorResources
from telemanom.utility import create_path
from telemanom.channel import Channel
from telemanom.errors import Errors
from telemanom.modeling import Model, LiteModel



def secondsToStr(t):
    return strftime("%H:%M:%S", gmtime(t))

class DetectorLite:
    def __init__(self, labels_path=None, results_path='results/', config_path='config.yaml'):

        #TODO- inserire commenti

        self.config = helpers.Config(config_path)
        self.labels_path = labels_path
        self.channel = None
        self.mode = 'convert_predict' #[test, convert, predict, convert_predict]
        self.labels_path = labels_path
        self.results_path = results_path
        self.TFLite_results = []
        self.TF_results = []
        self.chan_df = None
        self.stats = []


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


    def save_results(self, mode='w', stats=True, results=True):
        """
            Save stats and prediction
        """

        base_filename = str(self.config.model_architecture) + '_' + str(self.config.n_layers)
        if base_filename.startswith('ESN') and self.config.serialization == True:
            base_filename = base_filename + '_SER'

        if stats:
            filename = base_filename + '_stats'
            stats_df = pd.DataFrame(self.stats)
            stats_df.to_csv('results/{}.csv'.format(filename), mode=mode, index=False)

        if results:
            filename = base_filename + '_TFLite_results'
            tflite_results_df = pd.DataFrame(self.TFLite_results)
            tflite_results_df.to_csv('results/{}.csv'.format(filename), mode=mode, index=False)

            """
            filename = base_filename + '_TF_results'
            tf_results_df = pd.DataFrame(self.TF_results)
            tf_results_df.to_csv('results/{}.csv'.format(filename), mode=mode, index=False)
            """


    def run(self):

        for i, row in self.chan_df.iterrows():
            channel_name = row.chan_id
            print('{}- {}'.format(i, row.chan_id))
            stats = {'chan_id': row.chan_id}

            #create channel and load dataset
            self.channel = Channel(self.config, channel_name)
            self.channel.load_data()

            tfLite_model = LiteModel(self.config, self.channel)
            # load TF model
            tf_model = Model(self.config, channel_name, self.channel)

            if self.mode.startswith('convert'):

                #monitor resources while doing predictions on TensorFlow model
                with ThreadPoolExecutor() as executor:
                    monitor = MonitorResources()
                    mem_thread = executor.submit(monitor.measure_usage)

                    try:
                        tf_model.batch_predict(self.channel)
                        #self.TF_results.append(self.get_results(row))
                    finally:
                        monitor.keep_monitoring = False
                        monitor.calculate_avg()
                        stats['TF Prediction Time'] = secondsToStr(tf_model.prediction_time)
                        stats['avg CPU% during TF prediction'] = monitor.cpu
                        stats['avg RAM used during TF prediction'] = monitor.ram

                # monitor resources while doing converion
                with ThreadPoolExecutor() as executor:
                    monitor = MonitorResources()
                    mem_thread = executor.submit(monitor.measure_usage)
                    try:
                        # convert model to TF Lite
                        print('Convert to TensorFlow Lite')
                        tfLite_model.convert(tf_model.model)
                        print('Conversion completed (from {}Kb to {}Kb)'.format(tf_model.size, tfLite_model.size))
                    finally:
                        monitor.keep_monitoring = False
                        monitor.calculate_avg()
                        stats['conversion time'] = secondsToStr(tfLite_model.conversion_time)
                        stats['avg CPU% during conversion'] = monitor.cpu
                        stats['avg RAM used during conversion'] = monitor.ram
                        stats['size variation'] = [tf_model.size, tfLite_model.size]


            if self.mode.endswith('predict'):

                # monitor resources while doing predictions on TensorFlow Lite model
                with ThreadPoolExecutor() as executor:
                    monitor = MonitorResources()
                    mem_thread = executor.submit(monitor.measure_usage)
                    try:
                        # predict using TFLite Model
                        print('Predicting with TensorFlowLite model')
                        tfLite_model.batch_predict(self.channel)
                        self.TFLite_results.append(self.get_results(row))

                    finally:
                        monitor.keep_monitoring = False
                        monitor.calculate_avg()
                        stats['TFLite prediction time'] = secondsToStr(tfLite_model.prediction_time)
                        stats['avg CPU% during TFLite prediction'] = monitor.cpu
                        stats['avg RAM used during TFLite prediction'] = monitor.ram



            if self.mode == 'test':
                break

            self.stats.append(stats)
            break

        self.save_results()