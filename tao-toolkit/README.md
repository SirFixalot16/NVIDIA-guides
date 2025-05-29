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

Create and activate conda environment
```shell
conda create -n taotest python==3.8
conda activate taotest
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
export PATH="/path/to/taodeep/TensorRT-8.6.1.6/bin:$PATH"
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

## Tao model training

Example model training for yolov4, consult tao-toolkit pretrained models documnentation for other models:
```shell
cd /path/to/taotest/models
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/pretrained_object_detection/resnet18/files?redirect=true&path=resnet_18.hdf5' -O resnet_18.hdf5
mkdir ./yolov4_results
```

### Example: Data pre-processing for YOLO Object Detection

Tao-toolkit trains yolov4 not on the standard YOLO label format, but the kitti format. <br>
YOLO label format: <br>
| Class (int) | x-center (float) | y-center (float) | width (float) | height (float) |
| :---        |    :----:        |          :----:  | :----:        | ---:           |
| 0           | 0.56             | 0.72             | 0.22          | 0.67           |
| 1           | 0.54             | 0.55             | 0.52          | 0.24           |

<br>Kitti format: <br>
| Values | Name       | Description                                              |
|:---    | :----:     | ---:                                                     | 
| 1      | type       | Object type (Car, Van, Truck, etc.)                      |
| 1      | truncated  | Float 0-1 (truncated ratio)                              |
| 1      | occluded   | Integer (0=visible, 1=partly occluded, 2=fully occluded) |
| 1      | alpha      | Observation angle (-pi..pi)                              |
| 4      | bbox       | 2D bounding box (x1,y1,x2,y2) in pixels                  |
| 3      | dimensions | 3D dimensions (height, width, length) in meters          |
| 3      | location   | 3D location (x,y,z) in camera coordinates                |
| 1      | rotation_y | Rotation around Y-axis in camera coordinates             |

<br>Example: <br>
Fire 0 0 0 614 181 727 284 0 0 0 0 0 0 0 <br>
Smoke 0 0 0 123 456 789 012 0 0 0 0 0 0 0

## Deploying tao trained models using TensorRT (TRT < 10.0)

For TensorRT version lower than 10.0, some tao trained models may need TensorRT OSS for creating custom parser for Deepstream deployment. <br>
For TensorRT 8.6.1.6:
```shell
git clone -b master https://github.com/nvidia/TensorRT TensorRT -b release/8.6
cd TensorRT
git submodule update --init --recursive
```

Building TensorRT OSS:
```shell
CC=/usr/bin/gcc-10 \
CXX=/usr/bin/g++-10 \
CUDAHOSTCXX=/usr/bin/g++-10 \
cmake .. -DTRT_LIB_DIR=/path/to/taotest/TensorRT-8.6.1.6/lib -DTRT_BIN_DIR=/path/to/taotest/TensorRT_OSS/TensorRT/out
```

### Example: Setting up YOLOv4-tiny Object Detection
...
