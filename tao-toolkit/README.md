# Tao-toolkit
Guides for installation and usage of NVIDIA tools

## Requirements
<p>
<ul>
<li>Ubuntu: 20.04, 22.04 for Deepstream compatibility </li>
<li>CUDA: 12.x </li>
<li>TensorRT: 8.6.1.6 for CUDA 12.x</li>
<li>CuDNN: 8.9.7 for CUDA 12.x </li>
<li>Docker images: 
<ul>
<li>nvcr.io/nvidia/tao/tao-toolkit:5.0.0-deploy </li>
<li>nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5 </li>
</ul>
</li>
<li>Miniconda: Python 3.8.x </li>
<li>Storage: >120GB </li>
<li>GPU: VRAM>12GB </l>
</ul>
</p>


## Jupyter notebook installation
<p>
Check tao_colab.ipynb for example instructions for training YOLOV4 model.
</p>

## Desktop / Server installation

Make folder for storing installation files and model, data files <br>
```shell
mkdir /path/to/taotest
cd /path/to/taotest
mkdir models
mkdir data
```
Download TensorRT
```shell
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
tar -xzf ./TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz -C ./
ls
export TRT_LIB_PATH="/path/to/taotest/TensorRT-8.6.1.6/lib"
export TRT_INC_PATH="/path/to/taotest/TensorRT-8.6.1.6/include"
sudo cp /path/to/taotest/TensorRT-8.6.1.6/lib/* /usr/lib/x86_64-linux-gnu/
sudo cp /path/to/taotest/TensorRT-8.6.1.6/include/* /usr/include
```

Download CuDNN
```shell
wget https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.7/local_installers/12.x/cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz/
tar -xf ./cudnn-linux-x86_64-8.9.7.29_cuda12-archive.tar.xz -C ./
mv ./cudnn-linux-x86_64-8.9.7.29_cuda12-archive ./cudnn-8.9.7
sudo cp /path/to/taotest/cudnn-8.9.7/lib/* /usr/lib/x86_64-linux-gnu/
sudo cp /path/to/taotest/cudnn-8.9.7/include/* /usr/include
ls
```

Download and install NVIDIA TAO (remember to change "/path/to/", or change the bash script file manually)
```shell
git clone https://github.com/NVIDIA-AI-IOT/nvidia-tao.git
sed -i "s|PATH_TO_TRT|/path/to/taotest|g" /path/to/taotest/nvidia-tao/tensorflow/setup_env_desktop.sh
sed -i "s|TRT_VERSION|8.6.1.6|g" /path/to/taotest/nvidia-tao/tensorflow/setup_env_desktop.sh
sh /path/to/taotest/nvidia-tao/tensorflow/setup_env_desktop.sh
```

From here, basic operations for tao-toolkit can be used, but for training and deploying tao models. Beyond version 5.0.0, Docker images must be pulled
```shell
docker pull nvcr.io/nvidia/tao/tao-toolkit:5.0.0-tf1.15.5
docker pull nvcr.io/nvidia/tao/tao-toolkit:5.0.0-deploy
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
      "source": "/path/to/taotest",
      "destination": "/path/to/taotest",
      "type": "bind" 
    }]
}
```
Tao-toolkit version 5.0.0 is installed and all functions can be used
```shell
tao
```