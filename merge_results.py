from time import strftime
from time import gmtime
import os
import pandas as pd
import statistics
import sys

CSV_PATH = sys.argv[1]


def merge_stats(filename):
    merged = {}
    architectures = list(results.keys())

    for key in results[architectures[0]].keys():
        col = []
        for architecture in results:
            for value in results[architecture][key]:
                print(architecture, key, value)
                val = value
                print(type(val))
                if 'Time' in key and ':' in str(val):
                    vals = val.split(':')
                    val = 60*int(vals[1]) + int(vals[2])

                col.append(val)
        if key == 'chan_id':
            col.append(col[0])
        else:
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
    if 'stats' in filename:
        print(filename)
        architecture = filename[:-4]
        results[architecture] = pd.read_csv(CSV_PATH + filename)


results = merge_stats(list(results.keys())[0][:-4])
