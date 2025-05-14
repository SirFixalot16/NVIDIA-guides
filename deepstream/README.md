# Deepstream
Guides for installation and usage of NVIDIA tools

## Requirements
<p>
<ul>
<li>Ubuntu: 22.04 </li>
<li>CUDA: 12.6 </li>
<li>TensorRT: 10.3.0.26 for CUDA 12.x </li>
<li>CuDNN: 9.3.0 for CUDA 12.x </li>
<li>Docker images: 
<ul>
<li>nvcr.io/nvidia/tao/tao-toolkit:5.0.0-deploy </li>
<li>nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5 </li>
</ul>
</li>
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

Download TensorRT
```shell
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
tar -xzf ./TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz -C ./
ls
export TRT_LIB_PATH="/path/to/taodeep/TensorRT-10.3.0.26/lib"
export TRT_INC_PATH="/path/to/taodeep/TensorRT-10.3.0.26/include"
sudo cp /path/to/taodeep/TensorRT-10.3.0.26/lib/* /usr/lib/x86_64-linux-gnu/
sudo cp /path/to/taodeep/TensorRT-10.3.0.26/include/* /usr/include
```

Download CuDNN
```shell
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz
tar -xf ./cudnn-linux-x86_64-9.3.0.75_cuda12-archive.tar.xz -C ./
mv ./cudnn-linux-x86_64-9.3.0.75_cuda12-archive ./cudnn-9.3.0
sudo cp /path/to/taodeep/cudnn-9.3.0/lib/* /usr/lib/x86_64-linux-gnu/
sudo cp /path/to/taodeep/cudnn-9.3.0/include/* /usr/include
ls
```

Beyond version 5.0.0, Docker images must be pulled
```shell
docker pull nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5
docker pull nvcr.io/nvidia/tao/tao-toolkit:5.0.0-deploy
```

Log in to nvcr:
```shell
docker login nvcr.io
Username: $oauthtoken
Password: <MY API KEY>
```

Create .tao_mounts.json at home folder
```shell
nano /path/to/home/.tao_mounts.json
```
Enter the following:
```json
{
  "DockerOptions": {
    "user": "uid:gid"
  },
  "Mounts": [
    {
      "source": "/path/to/taodeep",
      "destination": "/path/to/taodeep",
      "type": "bind" 
    }]
}
```
Tao-toolkit version 5.0.0 is installed and all functions can be used
```shell
tao
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


## Tao model conversion to engine
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