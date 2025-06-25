# Deepstream-app
Guides for installation and usage of NVIDIA tools

## Requirements
<p>
<ul>
<li>Ubuntu: 22.04 </li>
<li>CUDA: 12.2 </li>
<li>TensorRT: 8.6.1.6 for CUDA 12.x </li>
<li>CuDNN: 8.9.7 for CUDA 12.x </li>
<li>Deepstream: 6.4 </li>
<li>Miniconda: Python 3.10.x </li>
<li>Storage: >120GB </li>
<li>GPU: VRAM>12GB </l>
</ul>
</p>

## Installation
Install python bindings according to this page: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git

## Configuration: Peoplenet
The intial configuration for running PeopleNet is based on deepstream_test_1 of the above repository, modified accordingly to match the included peoplenet.sh - which is the gstreamer command line to run peoplenet with gstreamer
```shell
conda deactivate
python3 peoplenet.py </path/to/video.mp4>
```

## Configuration: Peoplenet-Redaction
The intial configuration for running PeopleNet is based on deepstream_imagedata_multistream_redaction of the above repository.
```shell
conda deactivate
python3 peoplenet_multistream_redaction.py -i file:///path/to/video.mp4 -c H264
```

## Configuration: YOLO Face Object Detection + ArcFace
Running YOLO Face Objection Detection in Deepstream pipeline and ArcFace on Python pipeline, based on deepstream_imagedata_multistream_redaction of the above repository.
```shell
conda deactivate
python3 peoplenet_multistream_redaction.py -i file:///path/to/video.mp4 -c H264
```


## References
<ul>
<li>
https://github.com/Kojk-AI/deepstream-face-recognition
</li>
<li>
https://github.com/riotu-lab/deepstream-facenet/
</li>
<li>
https://forums.developer.nvidia.com/t/what-model-to-use-for-face-recognition/216693/22
</li>
</ul>