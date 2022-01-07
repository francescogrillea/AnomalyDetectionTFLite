import pandas as pd


def compare_csv(paths):

    files = []
    for p in paths:
        files.append(pd.read_csv(p))

    header = files[0].columns.values

    for i in range(header.size):
        print(header[i])
        for j in range(len(files)):
            print('\t{}: {}'.format(j, files[j].values[0][i]))



if __name__ == '__main__':
    paths = []
    paths.append('results/test.csv')
    paths.append('results/ESN_1_TFLite_results.csv')
    paths.append('results/LSTM_1_TF_results.csv')


    compare_csv(paths)