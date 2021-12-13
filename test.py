from telemanom import helpers
import pandas as pd

def compare_predictions(config_path='config.yaml', results_path='results/'):

    """
    Function to compare predictions made from the TF Model and the TFLite model
    :param config_path:
    :param results_path:
    :return:
    """
    config = helpers.Config(config_path)
    filename = str(config.model_architecture) + '_' + str(config.n_layers)
    if config.model_architecture == 'ESN' and config.serialization == True:
        filename = filename + '_SER'
    filename = '_predictions_' + filename + '.csv'

    TF_results = pd.read_csv(results_path+'TF'+filename)
    TFLite_results = pd.read_csv(results_path + 'TFLite' + filename)
    header = list(TF_results.columns.values)

    i = 0
    for a_row, b_row in zip(TF_results.iterrows(), TFLite_results.iterrows()):
        i += 1
        for a,b in zip(a_row[1], b_row[1]):
            if not a == b:
                print(header[i])
                print('\t{}'.format(a))
                print('\t{}'.format(b))



if __name__ == '__main__':
    compare_predictions()
