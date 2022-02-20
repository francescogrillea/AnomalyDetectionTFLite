import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# width of the bars
barWidth = 0.3

CSV_PATH = r"C:\Users\grill\OneDrive - University of Pisa\Anomaly Detection su JetsonNano\Risultati Sperimentali\JetsonNano\CPU\\"
IMG_PATH = r"C:\Users\grill\Desktop\aa\\"

DEINED_CHARS = [']','[',',','\'']

def plot_stats(title, unit, data, save=True, show=False):

    n = len(data['LSTM_1'])

    avg_bars = []
    std_dev_line = []
    labels = ['TF Model', 'TFLite Model']
    colors = ['blue', 'cyan']
    ticks = [0.15,1.15,2.15]

    for i in range(n):
        avg = []
        std_dev = []
        for architecture in data.keys():
            print(title, architecture, data[architecture])
            avg.append(data[architecture][i][0])
            std_dev.append(data[architecture][i][1])

        avg_bars.append(avg)
        std_dev_line.append(std_dev)

    if n == 3:
        labels.append('Conversion')
        colors.append('green')
        ticks = [0.3,1.3,2.3]
        avg_bars[1], avg_bars[2] = avg_bars[2], avg_bars[1]
        std_dev_line[1], std_dev_line[2] = std_dev_line[2], std_dev_line[1]

    # The x position of bars
    x = [np.arange(len(avg_bars[0]))]
    for i in range(1,n):
        val = [k + barWidth for k in x[i-1]]
        x.append(val)


    for i in range(n):
        plt.bar(x[i], avg_bars[i], width=barWidth, color=colors[i], edgecolor='black', yerr=std_dev_line[i], capsize=7, label=labels[i])

    x_labels = [architecture + 'L' for architecture in data.keys()]
    plt.xticks([r for r in range(len(avg_bars[0]))], x_labels)
    #plt.xticks(ticks, x_labels)

    plt.ylabel(unit)
    plt.legend()
    plt.title(title)
    if save:
        filename = title.split(' ')[0]
        plt.savefig(IMG_PATH+filename+'.png')
    if show:
        plt.show()
    plt.clf()




results = {}
#read .csv files
for filename in os.listdir(CSV_PATH):
    if filename.endswith('stats.csv'):
        architecture = filename[:-10]
        results[architecture] = pd.read_csv(CSV_PATH + filename).iloc[-1]

#print(results['LSTM_1'])


cpu_info = {}
ram_info = {}
time_info = {}
size_info = {}


#preprocess the input data
for architecture in results:
    cpu_info[architecture] = [] #TF Prediction, Conversion, TFLite Prediction
    ram_info[architecture] = [] #TF Prediction, Conversion, TFLite Prediction
    time_info[architecture] = [] #TF Prediction, Conversion, TFLite Prediction
    size_info[architecture] = [] #TF, TFLite

    for i, key in zip(range(1,30), results[architecture].keys()[1:]):

        value = results[architecture][key]

        for c in DEINED_CHARS:
            value = value.replace(c, '')
        if i in [1,4,9]:
            values = value.split(' ')
            one = values[0].split(':')
            two = values[1].split(':')

            v_one = float(one[-1]) + (60 * int(one[-2]))
            v_two = int(two[-1]) + (60 * int(two[-2]))
            values = [v_one, v_two]
        else:
            values = [float(x) for x in value.split(' ')]

        if 'CPU' in key:
            cpu_info[architecture].append(values)
        elif 'RAM' in key:
            ram_info[architecture].append(values)
        elif 'Time' in key:
            time_info[architecture].append(values)
        elif 'size' in key:
            size_info[architecture].append(values)

plot_stats('CPU Usage', '%', cpu_info)
plot_stats('RAM Usage', 'MB', ram_info)
plot_stats('Time Elapsed', 'sec', time_info)
plot_stats('Size Variation', 'KB', size_info)


