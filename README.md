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
</ul>
