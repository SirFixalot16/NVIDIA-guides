# NVIDIA-guides
Guides for installation and usage of NVIDIA tools

## Nvidia frameworks compatibility
Proposed compatibility matrix for python 3.8 - 3.10 on Ubuntu 22.04:
<br> 
| CUDA | TensorRT   | CuDNN  | tao-toolkit    | Deepstream |
| :--- |  :----:    | :----: | :----:         | ---:       |
| 12.x | 8.6.1.6    | 8.9.7  | 5.0.0-tf1.15.5 | 6.4        |
| 12.6 | >10.3.0.26 | 9.3.0  | 5.0.0-tf1.15.5 | 7.1        |

Proposed installation process:
<br>
Miniconda -> CUDA (1 version) -> CuDNN (1 version) -> gstreamer (disable conda) -> Deepstream (1 version) -> TensorRT (multiple, enable conda) -> tao-toolkit (1 version)

### CUDA version change
If CUDA version has official uninstaller:
```shell
ls -l /usr/local/cuda-*/bin/
sudo /usr/local/cuda-12.9/bin/cuda-uninstaller
```

If not:
```shell
sudo apt-get --purge remove cuda
sudo apt-get autoremove
dpkg --list |grep "^rc" | cut -d " " -f 3 | xargs sudo dpkg --purge
```
After installation, export path with:
```shell
export PATH=${PATH}:/usr/local/cuda-x/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-x
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-x/lib64
```

### CuDNN version change 
```shell
rm -f /usr/include/cudnn.h
rm -f /usr/lib/x86_64-linux-gnu/*libcudnn*
rm -f /usr/local/cuda-*/lib64/*libcudnn*

cp -P /path/to/new/cudnn/include/cudnn.h /usr/include
cp -P /path/to/new/cudnn/lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
```

### Libraries to be installed manually:

<ul>
<li>
DeepStream-Yolo: https://github.com/marcoslucianops/DeepStream-Yolo.git
</li>
<li>
TensorRT / CuDNN: .tar instead of .deb for ease of configurations
</li>
<li>
tao-toolkit (tao-tensorflow): https://github.com/NVIDIA-AI-IOT/nvidia-tao.git
</li>
<li>
tao-pytorch: https://github.com/NVIDIA/tao_pytorch_backend
</li>  
<li>
Deepstream-tao-apps: https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps/tree/master
</li>
<li>
TensorRT OSS (for TensorRT version below 10): https://github.com/NVIDIA/TensorRT/tree/release/7.0
</li>
<li>
Extended TensorRT operations: https://github.com/wang-xinyu/tensorrtx
</li>
</ul>

## References
<ul>
<li>
https://stackoverflow.com/questions/50213021/best-practice-for-upgrading-cuda-and-cudnn-for-tensorflow
</li>
<li>
CUDA installation guide: https://docs.nvidia.com/cuda/archive/9.1/cuda-installation-guide-linux/index.html
</li>
<li>
https://github.com/zhouyuchong/face-recognition-deepstream
</li>
<li>
https://github.com/SauravSinghPaliwal/Deepstream-Face-Recognition
</li>
</ul>
