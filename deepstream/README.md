# Deepstream
Guides for installation and usage of NVIDIA tools

## Requirements
<p>
<ul>
<li>Ubuntu: 22.04 </li>
<li>CUDA: 12.6 </li>
<li>TensorRT: 10.11.0.33 for CUDA 12.x </li>
<li>CuDNN: >9.3.0 for CUDA 12.x </li>
<li>Miniconda: Python 3.10.x </li>
<li>Storage: >120GB </li>
<li>GPU: VRAM>12GB </l>
</ul>
</p>

## Desktop / Server installation

Make folder for storing installation files and model, data files <br>
```shell
mkdir /path/to/taodeep
cd /path/to/taodeep
mkdir models
mkdir data
```

Create and activate conda environment
```shell
conda create -n taodeep python==3.10.15
conda activate taodeep
```

Download TensorRT (tarball)
```shell
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/tars/TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz
tar -xzf ./TensorRT-10.11.0.33.Linux.x86_64-gnu.cuda-12.9.tar.gz -C ./
ls
export TRT_LIB_PATH="/path/to/taodeep/TensorRT-10.11.0.33/lib"
export TRT_INC_PATH="/path/to/taodeep/TensorRT-10.11.0.33/include"
sudo cp /path/to/taodeep/TensorRT-10.11.0.33/lib/* /usr/lib/x86_64-linux-gnu/
sudo cp /path/to/taodeep/TensorRT-10.11.0.33/include/* /usr/include
sudo cp /path/to/taodeep/TensorRT-10.11.0.33/lib/* /usr/local/cuda/lib64
sudo cp /path/to/taodeep/TensorRT-10.11.0.33/include/* /usr/local/cuda/include
export PATH="/path/to/taodeep/TensorRT-10.11.0.33/bin:$PATH"
```

Download TensorRT (.deb)
```shell
wget https://developer.download.nvidia.com/compute/tensorrt/10.11.0/local_installers/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.11.0-cuda-12.9/nv-tensorrt-local-5BF87A98-keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install libnvinfer10 libnvinfer-dev libnvinfer-plugin10 libnvinfer-plugin-dev libnvonnxparsers10 libnvonnxparsers-dev
```

Download CuDNN (tarball)
```shell
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz
tar -xf ./cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz -C ./
mv ./cudnn-linux-x86_64-9.3.0.75_cuda12-archive ./cudnn-9.3.0
export CUDNN_INC_DIR=/path/to/taodeep/cudnn-9.3.0/include
export CUDNN_LIB_DIR=/path/to/taodeep/cudnn-9.3.0/lib
sudo cp /path/to/taodeep/cudnn-9.3.0/lib/* /usr/lib/x86_64-linux-gnu/
sudo cp /path/to/taodeep/cudnn-9.3.0/include/* /usr/include
sudo cp /path/to/taodeep/cudnn-9.3.0/lib/* /usr/local/cuda/lib64
sudo cp /path/to/taodeep/cudnn-9.3.0/include/* /usr/local/cuda/include
ls
```

Download CuDNN (.deb)
```shell
wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.3.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn-cuda-12
```

## Deepstream

Deepstream prerequisites:
```shell
sudo apt install libssl3 libssl-dev libgles2-mesa-dev libgstreamer1.0-0 gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-0 libjansson4 libyaml-cpp-dev libjsoncpp-dev protobuf-compiler gcc make git python3
```

Deepstream deinstallation in case of faulty installation:
```shell
sudo rm -rf /usr/local/deepstream /usr/lib/x86_64-linux-gnu/gstreamer-1.0/libgstnv* /usr/bin/deepstream* /usr/lib/x86_64-linux-gnu/gstreamer-1.0 libnvdsgst* /usr/lib/x86_64-linux-gnu/gstreamer-1.0/deepstream* /opt/nvidia/deepstream/deepstream*
sudo rm -rf /usr/lib/x86_64-linux-gnu/libv41/plugins/libcuvidv4l2_plugin.so
sudo dpkg --remove --force-remove-reinstreq deepstream-7.1
sudo dpkg --configure -a
sudo apt --fix-broken install
sudo apt-get autoremove
sudo apt-get clean
```

Installing Deepstream (.deb):
```shell
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.1/files?redirect=true&path=deepstream-7.1_7.1.0-1_amd64.deb' -o 'deepstream-7.1_7.1.0-1_amd64.deb'
sudo apt install ./deepstream-7.1_7.1.0-1_amd64.deb
```
Installing Deepstream (tarball):
```shell
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.1/files?redirect=true&path=deepstream_sdk_v7.1.0_x86_64.tbz2' -o 'deepstream_sdk_v7.1.0_x86_64.tbz2'
sudo tar -xvf deepstream_sdk_v7.1.0_x86_64.tbz2 -C /
cd /opt/nvidia/deepstream/deepstream-7.1
sudo ./install.sh
sudo ldconfig
```

Test run
```shell
deepstream-app
```


## Model conversion to engine
.

## Running engine on deepstream

Sample deepstream config file for YOLO inference:
```shell
```

Sample config file for YOLO engine:
```shell
```

Sample config file for labels:
```shell
```
