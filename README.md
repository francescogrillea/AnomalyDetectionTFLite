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
Architecture aarch64 \
OS Raspbian 11 (bullseye) \
Python v3.9.2 \
Pip v20.3.4


## Install
CPU Version [TF v2.7.0] \
`pip install tensorflow -f https://tf.kmtea.eu/whl/stable.html` \
or if needed a specify TF version \
`pip install tensorflow==2.7.0 -f https://tf.kmtea.eu/whl/stable.html` \
`sudo pip install -r requirements`

## Known Problems
* ESN 2L: TFLite prediction killed at 0% cause 1.05GB of swap memory is not enough (Jetson has 2G of swap memory)

