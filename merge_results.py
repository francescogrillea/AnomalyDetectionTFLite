from time import strftime
from time import gmtime
import os
import pandas as pd
import statistics

CSV_PATH = r"C:\Users\grill\OneDrive - University of Pisa\Anomaly Detection su JetsonNano\Risultati Sperimentali\JetsonNano\CPU\ESN 2\\"



def merge_stats(filename):
    merged = {}

    for key in results['ESN_2'].keys():
        if key == 'chan_id':
            continue
        col = []
        for architecture in results:
            for value in results[architecture][key]:
                val = value
                if 'Time' in key and ':' in str(val):
                    vals = val.split(':')
                    val = 60*int(vals[1]) + int(vals[2])

                col.append(val)
        #calculate avg and stddev
        avg = round(statistics.mean(col), 2)
        stdev = round(statistics.stdev(col), 2)
        if 'Time' in key:
            avg = strftime("%H:%M:%S", gmtime(avg))
            stdev = strftime("%H:%M:%S", gmtime(stdev))
            #trasforma in 00:00:00
            for t in range(len(col)):
                col[t] = strftime("%H:%M:%S", gmtime(col[t]))

        col.append([avg, stdev])
        merged[key] = col


    merged_df = pd.DataFrame(merged)
    merged_df.to_csv('results/{}.csv'.format(filename), index=False)
    return merged


#----------------------------#
results = {}
#read .csv files
for filename in os.listdir(CSV_PATH):
    if filename.endswith('stats.csv'):
        print(filename)
        architecture = filename[:-10]
        results[architecture] = pd.read_csv(CSV_PATH + filename)

#print(results['LSTM_1'])

results = merge_stats('merged_stats')
