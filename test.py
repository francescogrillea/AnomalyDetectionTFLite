import datetime

import numpy as np
import pandas as pd

def compare_predictions(config_path='config.yaml', results_path_a='results/', results_path_b= 'results/'):

    tf_file = pd.read_csv(results_path_a)
    tfLite_file = pd.read_csv(results_path_b)
    tf_values = tf_file.values[0]
    tfLite_values = tfLite_file.values[0]
    header = tf_file.columns.values

    for i in range(header.size):
        print(header[i])
        print('\tTF : {}'.format(tf_values[i]))
        print('\tTFL: {}'.format(tfLite_values[i]))



if __name__ == '__main__':
    a_path = 'results/LSTM_1_TF_results.csv'
    b_path = 'results/LSTM_1_TFLite_results.csv'
    compare_predictions(results_path_a=a_path, results_path_b=b_path)