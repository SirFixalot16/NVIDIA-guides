gst-launch-1.0 -v filesrc location=/home/khmt/taotest/arcface/tvt.mp4 ! \
    qtdemux name=demux ! \
    queue ! \
    h264parse ! \
    nvv4l2decoder ! \
    queue ! \
    nvvideoconvert ! \
    "video/x-raw(memory:NVMM), format=NV12, width=1280, height=720" ! \
    mux.sink_0 nvstreammux name=mux batch-size=1 width=1280 height=720 live-source=0 ! \
    queue ! \
    nvdsosd ! \
    nvvideoconvert ! \
    "video/x-raw, format=RGBA" ! \
    nveglglessink sync=false