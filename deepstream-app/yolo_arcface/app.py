import argparse
import sys
import paddle
import cv2
import numpy as np
from paddle.inference import Config, create_predictor
sys.path.append('../')
import gi
import configparser

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
from ctypes import *
import time
import math
import platform
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
from common.FPS import PERF_DATA
import pyds
import os
import os.path
from os import path
from face_recog import Face_Recog
perf_data = None
frame_count = {}
saved_count = {}

global PGIE_CLASS_ID_FACE
PGIE_CLASS_ID_FACE = 0

MAX_DISPLAY_LEN = 64
MUXER_OUTPUT_WIDTH = 720
MUXER_OUTPUT_HEIGHT = 576
MUXER_BATCH_TIMEOUT_USEC = 33000
TILED_OUTPUT_WIDTH = 720  
TILED_OUTPUT_HEIGHT = 576
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["Face"]
HIDE_LABELS = {"face"}  # những nhãn muốn ẩn

MIN_CONFIDENCE = 0.3
MAX_CONFIDENCE = 0.4



def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = {PGIE_CLASS_ID_FACE: 0}

        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1

            if obj_meta.class_id == PGIE_CLASS_ID_FACE:
                obj_meta.rect_params.border_width = 1
                obj_meta.rect_params.has_bg_color = 0
                obj_meta.rect_params.bg_color.red = 0.0
                obj_meta.rect_params.bg_color.green = 0.0
                obj_meta.rect_params.bg_color.blue = 0.0
                obj_meta.rect_params.bg_color.alpha = 0.0

            if saved_count["stream_{}".format(frame_meta.pad_index)] % 1 == 0:
                if is_first_obj:
                    is_first_obj = False
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    print(f'Frame number: {frame_number}')
                    # crop_img, image_path, similarity, label = crop_object(n_frame, obj_meta)
                    crop_img, image_path, similarity, label  = face_recog.crop_object(n_frame,obj_meta)


        
                    if image_path is not None and label != "Unknown":
                        display_label = "" if label.lower() in HIDE_LABELS else label
                        obj_meta.text_params.display_text = f"{display_label}, S={similarity:.2f}" if display_label else f"S={similarity:.2f}"


                        obj_meta.rect_params.border_width = 1
                    else:
                        obj_meta.text_params.display_text = "Unknown"
                    obj_meta.text_params.font_params.font_size = 10
                    obj_meta.text_params.x_offset = int(obj_meta.rect_params.left)
                    # obj_meta.text_params.y_offset = int(obj_meta.rect_params.top - 10)

                    frame_copy = np.array(crop_img, copy=True, order='C')
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_RGBA2BGRA)
                    if platform_info.is_integrated_gpu():
                        pyds.unmap_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)

                    save_image = True

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        stream_index = "stream{0}".format(frame_meta.pad_index)
        perf_data.update_fps(stream_index)
        if save_image:
            img_path = f"{folder_name}/stream_{frame_meta.pad_index}/frame_{frame_number}.jpg"
            cv2.imwrite(img_path, frame_copy)
        saved_count["stream_{}".format(frame_meta.pad_index)] += 1
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    if gstname.find("video") != -1:
        if features.contains("memory:NVMM"):
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write("Error: Decodebin did not pick nvidia decoder plugin.\n")

def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)
    if not platform_info.is_integrated_gpu() and name.find("nvv4l2decoder") != -1:
        Object.set_property("cudadec-memtype", 2)

def create_source_bin(index, uri):
    print("Creating source bin")
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write("Unable to create source bin\n")

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write("Unable to create uri decode bin\n")
    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write("Failed to add ghost pad in source bin\n")
        return None
    return nbin

def main(uri_inputs, codec, bitrate):
    number_sources = len(uri_inputs)
    global perf_data, folder_name, platform_info
    perf_data = PERF_DATA(number_sources)
    folder_name = "out_crops"
    os.makedirs(folder_name, exist_ok=True)
    print("Frames will be saved in ", folder_name)
    platform_info = PlatformInfo()
    Gst.init(None)

    print("Creating Pipeline\n")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
    print("Creating streamux\n")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux\n")

    pipeline.add(streammux)
    for i in range(number_sources):
        os.makedirs(folder_name + "/stream_" + str(i), exist_ok=True)
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, "\n")
        uri_name = uri_inputs[i]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin\n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.request_pad_simple(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin\n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin\n")
        srcpad.link(sinkpad)

    print("Creating Pgie\n")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie\n")

    print("Creating nvvidconv1\n")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write("Unable to create nvvidconv1\n")
    print("Creating filter1\n")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write("Unable to get the caps filter1\n")
    filter1.set_property("caps", caps1)
    print("Creating tiler\n")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("Unable to create tiler\n")
    print("Creating nvvidconv\n")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv\n")
    print("Creating nvosd\n")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd\n")
    nvosd.set_property("display-text", 1)  # Enable text display
    nvosd.set_property("display-bbox", 1)  # Keep bounding box
    nvvidconv_postosd = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd")
    if not nvvidconv_postosd:
        sys.stderr.write("Unable to create nvvidconv_postosd\n")

    caps = Gst.ElementFactory.make("capsfilter", "filter")
    caps.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))

    if codec == "H264":
        encoder = Gst.ElementFactory.make("nvv4l2h264enc", "encoder")
        rtppay = Gst.ElementFactory.make("rtph264pay", "rtppay")
        print("Creating H264 Encoder and rtppay")
    elif codec == "H265":
        encoder = Gst.ElementFactory.make("nvv4l2h265enc", "encoder")
        rtppay = Gst.ElementFactory.make("rtph265pay", "rtppay")
        print("Creating H265 Encoder and rtppay")
    if not encoder or not rtppay:
        sys.stderr.write("Unable to create encoder or rtppay")

    encoder.set_property('bitrate', bitrate)
    if platform_info.is_integrated_gpu():
        encoder.set_property('preset-level', 1)
        encoder.set_property('insert-sps-pps', 1)

    updsink_port_num = 5400
    sink = Gst.ElementFactory.make("udpsink", "udpsink")
    if not sink:
        sys.stderr.write("Unable to create udpsink")

    sink.set_property('host', '224.224.255.255')
    sink.set_property('port', updsink_port_num)
    sink.set_property('async', False)
    sink.set_property('sync', 1)

    print("Playing file {} ".format(uri_inputs))

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    pgie.set_property('config-file-path', "config_face.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
              number_sources, "\n")
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    if not platform_info.is_integrated_gpu():
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)
        nvvidconv_postosd.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline\n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    pipeline.add(nvvidconv_postosd)
    pipeline.add(caps)
    pipeline.add(encoder)
    pipeline.add(rtppay)
    pipeline.add(sink)

    print("Linking elements in the Pipeline\n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(nvvidconv_postosd)
    nvvidconv_postosd.link(caps)
    caps.link(encoder)
    encoder.link(rtppay)
    rtppay.link(sink)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    rtsp_port_num = 8554
    server = GstRtspServer.RTSPServer.new()
    server.props.service = "%d" % rtsp_port_num
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        "( udpsrc name=pay0 port=%d buffer-size=524288 caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)%s, payload=96 \" )"
        % (updsink_port_num, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory("/ds-test", factory)

    print("\n *** DeepStream: Launched RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n" % rtsp_port_num)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write("Unable to get sink pad\n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    print("Starting pipeline\n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)

def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Output Sample Application Help ')
    parser.add_argument("-i", "--uri_inputs", metavar='N', type=str, nargs='+',
                        help='Path to inputs URI e.g. rtsp:// ... or file:// separated by space')
    parser.add_argument("-c", "--codec", default="H264",
                        help="RTSP Streaming Codec H264/H265, default=H264", choices=['H264', 'H265'])
    parser.add_argument("-b", "--bitrate", default=4000000,
                        help="Set the encoding bitrate", type=int)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    print("URI Inputs: " + str(args.uri_inputs))
    return args.uri_inputs, args.codec, args.bitrate

if __name__ == '__main__':
    model_dir = '/path/to/arcface_iresnet50_v1.0_infer'
    face_recog = Face_Recog('/path/to/arcface_iresnet50_v1.0_infer')
    model = face_recog.load_model_predict()
    uri_inputs, out_codec, out_bitrate = parse_args()
    sys.exit(main(uri_inputs, out_codec, out_bitrate))