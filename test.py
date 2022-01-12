import pandas as pd


def compare_csv(paths):

    dataframes = []
    for p in paths:
        dataframes.append(pd.read_csv(p))

    header = dataframes[0].columns.values
    n_columns = dataframes[0].iloc[0].size
    n_rows = len(dataframes[0])

    for i in range(n_rows):
        print('Channel {}'.format(dataframes[0].iloc[i][0]))

        for k in range(1, len(header)):
            print('\t{}'.format(header[k]))
            for j in range(len(dataframes)):
                row = dataframes[j].iloc[i]
                print('\t\t{}'.format(row[k]))

        print('==========================================')



if __name__ == '__main__':
    paths = []
    paths.append('results/LSTM_1_TFLite_results.csv')
    paths.append('results/LSTM_1_TF_results.csv')


    compare_csv(paths)