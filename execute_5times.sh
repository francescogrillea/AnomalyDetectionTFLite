#!/bin/bash

TIMES=5

for i in $(seq 1 $TIMES); do
  echo "###############"
  echo "Iteration ${i} / ${TIMES}"
  echo "###############"
  echo ""
  python example.py -l labeled_anomalies.csv
  echo ""
  echo ""
  echo "";
done

python merge_results.py results/
rm results/*_it*