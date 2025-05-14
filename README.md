# NVIDIA-guides
Guides for installation and usage of NVIDIA tools

## Nvidia frameworks compatibility
Proposed compatibility matrix for python 3.8 - 3.10 on Ubuntu 22.04:
<br> 
| CUDA | TensorRT  | CuDNN  | tao-toolkit    | Deepstream |
| :--- |  :----:   | :----: | :----:         | ---:       |
| 12.x | 8.6.1.6   | 8.9.2  | 5.0.0-tf1.15.5 | 7.1        |
| 12.6 | 10.3.0.26 | 9.3.0  | 5.0.0-tf1.15.5 | 7.1        |

Proposed installation process:
<br>
Miniconda -> CUDA -> TensorRT -> tao-toolkit -> Deepstream


### Libraries to be installed manually:

<ul>
<li>
DeepStream-Yolo: https://github.com/marcoslucianops/DeepStream-Yolo.git
</li>
<li>
TensorRT / CuDNN: .tar instead of .deb for ease of configurations
</li>
<li>
tao-toolkit: https://github.com/NVIDIA-AI-IOT/nvidia-tao.git
</li>
</ul>