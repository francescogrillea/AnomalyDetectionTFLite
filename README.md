# Local

## Env

OS Windows Subsystem Linux on Windows 11 \
Python v3.8.10 \
Pip v22.0.3

## Install 

`pip install tensorflow==2.7.0` \
`pip install -r requirements`

## Execute

`python example.py -l labeled_anomalies.csv`

# NVIDIA Jetson Nano

## Env

OS Ubuntu 18.04 \
Python v3.6.9 \
Pip v21.3.1 \
JetPack v4.5 \
CUDA v10.2 \
CUDNN v8.0.0

## Install 

### CPU Version
`pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html`

### GPU Version
`sudo pip3 install -U h5py==3.1.0` \
`sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow` \
`sudo pip3 install -r requirements`

## Execute

`OPENBLAS_CORETYPE=ARMV8 python3 example.py -l labeled_anomalies.csv`


# Raspberry Pi3 (Model B)

## Env
Raspbian 9 \
Python v3.7.4 \
Pip v22.0.3


## Install - Method 1
install h5py package \
`$ sudo apt-get install libhdf5-dev`

Install TensorFlow v2.4.0

`$ pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl`

### Problems
TF Prediction: Ok \
Conversion: Segmentation Fault \
TFLite Prediction: cannot load .tflite model

## Install - Method 2
