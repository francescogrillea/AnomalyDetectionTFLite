# AnomalyDetectionTFLite
Neural Networks for anomaly detection on timeseries using TensorFlow and TensorFlow Lite.

## Install

### Ubuntu 20.04
`pip install tensorflow==2.7.0` \
`pip install -r requirements`

### Jetson Nano

##### CPU-based
`pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html` \
`pip install -r requirements`

##### GPU-based
`sudo pip3 install -U h5py==3.1.0` \
`sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow` 
`sudo pip3 install -r requirements`


### Raspberry Pi3 (Model B)


## Execute
`chmod +x execute_5times.sh` \
`./execute_5times.sh`
