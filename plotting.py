import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# width of the bars
barWidth = 0.3


HEADER = ['chan_id',
          'TF_Prediction_Time',
          'TF_Prediction_CPU',
          'TF_Prediction_RAM',
          'Conversion_Time',
          'Conversion_CPU',
          'Conversion_RAM',
          'TF_Size',
          'TFLite_Size',
          'TFLite_Prediction_Time',
          'TFLite_Prediction_CPU',
          'TFLite_Prediction_RAM',
          ]

CSV_PATH = 'results/'
IMG_PATH = 'results/imgs/'
DEINED_CHARS = [']','[',',','\'']


def plot_CPU_usage():
    avg_bar1 = []
    stdev_line1 = []

    avg_bar2 = []
    stdev_line2 = []

    avg_bar3 = []
    stdev_line3 = []

    # Plot CPU
    for architecture in results:
        avg_bar1.append(results[architecture][2][0])
        stdev_line1.append(results[architecture][2][1])

        avg_bar2.append(results[architecture][10][0])
        stdev_line2.append(results[architecture][10][1])

        avg_bar3.append(results[architecture][5][0])
        stdev_line3.append(results[architecture][5][1])

    # The x position of bars
    r1 = np.arange(len(avg_bar1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, avg_bar1, width=barWidth, color='blue', edgecolor='black', yerr=stdev_line1, capsize=7,
            label='TF Prediction')
    plt.bar(r2, avg_bar2, width=barWidth, color='cyan', edgecolor='black', yerr=stdev_line2, capsize=7,
            label='TFLite Prediction')
    plt.bar(r3, avg_bar3, width=barWidth, color='green', edgecolor='black', yerr=stdev_line3, capsize=7,
            label='Conversion')

    x_labels = [k + 'L' for k in results.keys()]
    y_label = 'CPU% Usage'

    plt.xticks([r for r in range(len(avg_bar1))], x_labels)
    plt.ylabel(y_label)
    plt.legend()
    plt.title('CPU Usage')
    plt.savefig('results/cpu.png')
    plt.show()

def plot_RAM_usage():
    avg_bar1 = []
    stdev_line1 = []

    avg_bar2 = []
    stdev_line2 = []

    avg_bar3 = []
    stdev_line3 = []

    # Plot CPU
    for architecture in results:
        avg_bar1.append(results[architecture][3][0])
        stdev_line1.append(results[architecture][3][1])

        avg_bar2.append(results[architecture][11][0])
        stdev_line2.append(results[architecture][11][1])

        avg_bar3.append(results[architecture][6][0])
        stdev_line3.append(results[architecture][6][1])

    # The x position of bars
    r1 = np.arange(len(avg_bar1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    plt.bar(r1, avg_bar1, width=barWidth, color='blue', edgecolor='black', yerr=stdev_line1, capsize=7,
            label='TF Prediction')
    plt.bar(r2, avg_bar2, width=barWidth, color='cyan', edgecolor='black', yerr=stdev_line2, capsize=7,
            label='TFLite Prediction')
    plt.bar(r3, avg_bar3, width=barWidth, color='green', edgecolor='black', yerr=stdev_line3, capsize=7,
            label='Conversion')

    x_labels = [k + 'L' for k in results.keys()]
    y_label = 'RAM usage (Kb)'

    plt.xticks([r for r in range(len(avg_bar1))], x_labels)
    plt.ylabel(y_label)
    plt.legend()
    plt.title('RAM Usage')
    plt.savefig('results/ram.png')
    plt.show()

def plot_size_comparison():
    avg_bar1 = []
    stdev_line1 = []

    avg_bar2 = []
    stdev_line2 = []

    # Plot CPU
    for architecture in results:
        avg_bar1.append(results[architecture][7][0])
        stdev_line1.append(results[architecture][7][1])

        avg_bar2.append(results[architecture][8][0])
        stdev_line2.append(results[architecture][8][1])

    # The x position of bars
    r1 = np.arange(len(avg_bar1))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, avg_bar1, width=barWidth, color='blue', edgecolor='black', yerr=stdev_line1, capsize=7,
            label='TF Model Size')
    plt.bar(r2, avg_bar2, width=barWidth, color='cyan', edgecolor='black', yerr=stdev_line2, capsize=7,
            label='TFLite Model Size')

    x_labels = [k + 'L' for k in results.keys()]
    y_label = 'Model Size (Kb)'

    plt.xticks([r for r in range(len(avg_bar1))], x_labels)
    plt.ylabel(y_label)
    plt.legend()

    plt.title('Model Size Comparison')
    plt.savefig('results/size.png')
    plt.show()

def plot_time_comparison():
    avg_bar1 = []
    stdev_line1 = []

    avg_bar2 = []
    stdev_line2 = []

    # Plot CPU
    for architecture in results:
        avg_bar1.append(results[architecture][1][0])
        stdev_line1.append(results[architecture][1][1])

        avg_bar2.append(results[architecture][9][0])
        stdev_line2.append(results[architecture][9][1])

    # The x position of bars
    r1 = np.arange(len(avg_bar1))
    r2 = [x + barWidth for x in r1]

    plt.bar(r1, avg_bar1, width=barWidth, color='blue', edgecolor='black', yerr=stdev_line1, capsize=7,
            label='TF Prediction')
    plt.bar(r2, avg_bar2, width=barWidth, color='cyan', edgecolor='black', yerr=stdev_line2, capsize=7,
            label='TFLite prediction')

    x_labels = [k + 'L' for k in results.keys()]
    y_label = 'Time Elapsed (s)'

    plt.xticks([r for r in range(len(avg_bar1))], x_labels)
    plt.ylabel(y_label)
    plt.legend()

    plt.title('Execution Time Comparison')
    plt.savefig('results/time.png')
    plt.show()


results = {}
#read .csv files
for filename in os.listdir(CSV_PATH):
    if filename.endswith('stats.csv'):
        architecture = filename[:-10]
        results[architecture] = pd.read_csv(CSV_PATH + filename).iloc[-1]

#preprocess the input data
for architecture in results:
    for i in range(1, len(results[architecture])):
        value = results[architecture][i]
        for c in DEINED_CHARS:
            value = value.replace(c, '')
        if i in [1,4,9]:
            values = value.split(' ')
            one = values[0].split(':')
            two = values[1].split(':')

            v_one = int(one[-1]) + (60 * int(one[-2]))
            v_two = int(two[-1]) + (60 * int(two[-2]))
            values = [v_one, v_two]
        else:
            values = [float(x) for x in value.split(' ')]
        results[architecture][i] = values

plot_CPU_usage()
plot_RAM_usage()
plot_size_comparison()
plot_time_comparison()