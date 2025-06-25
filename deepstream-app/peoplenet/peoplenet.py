#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.bus_call import bus_call
import pyds

PGIE_CLASS_ID_PERSON = 0
PGIE_CLASS_ID_BAG = 1
PGIE_CLASS_ID_FACE = 2
MUXER_BATCH_TIMEOUT_USEC = 33000

def osd_sink_pad_buffer_probe(pad, info, u_data):
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

        obj_counter = {
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BAG: 0,
            PGIE_CLASS_ID_FACE: 0
        }
        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.8)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        py_nvosd_text_params.display_text = (
            f"Frame Number={frame_number} Number of Objects={num_rects} "
            f"Face_count={obj_counter[PGIE_CLASS_ID_FACE]} Person_count={obj_counter[PGIE_CLASS_ID_PERSON]}"
        )
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
        py_nvosd_text_params.set_bg_clr = 1
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

def cb_newpad(demux, pad, data):
    """Callback function for qtdemux pad-added signal."""
    pipeline, h264parser = data
    caps = pad.get_current_caps()
    if not caps:
        return

    structure = caps.get_structure(0)
    if structure.get_name() == "video/x-h264":
        print("Linking qtdemux video pad to h264parse")
        parser_sinkpad = h264parser.get_static_pad("sink")
        if not parser_sinkpad:
            sys.stderr.write("Unable to get sink pad of h264parse\n")
            return
        if pad.link(parser_sinkpad) != Gst.PadLinkReturn.OK:
            sys.stderr.write("Failed to link qtdemux video pad to h264parse\n")

def main(args):
    if len(args) != 2:
        sys.stderr.write("usage: %s <media file or uri>\n" % args[0])
        sys.exit(1)

    platform_info = PlatformInfo()
    Gst.init(None)

    print("Creating Pipeline \n")
    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Unable to create Pipeline \n")
        sys.exit(1)

    source = Gst.ElementFactory.make("filesrc", "file-source")
    if not source:
        sys.stderr.write("Unable to create Source \n")
        sys.exit(1)

    demux = Gst.ElementFactory.make("qtdemux", "qt-demux")
    if not demux:
        sys.stderr.write("Cannot create demux \n")
        sys.exit(1)

    h264parser = Gst.ElementFactory.make("h264parse", "h264-parser")
    if not h264parser:
        sys.stderr.write("Unable to create h264 parser \n")
        sys.exit(1)

    decoder = Gst.ElementFactory.make("nvv4l2decoder", "nvv4l2-decoder")
    if not decoder:
        sys.stderr.write("Unable to create Nvv4l2 Decoder \n")
        sys.exit(1)

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux \n")
        sys.exit(1)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create pgie \n")
        sys.exit(1)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write("Unable to create nvvidconv \n")
        sys.exit(1)

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd \n")
        sys.exit(1)

    if platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64():
        print("Creating nv3dsink \n")
        sink = Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    else:
        print("Creating EGLSink \n")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write("Unable to create sink \n")
        sys.exit(1)

    print("Playing file %s " % args[1])
    source.set_property('location', args[1])
    if os.environ.get('USE_NEW_NVSTREAMMUX') != 'yes':
        streammux.set_property('width', 1920)
        streammux.set_property('height', 1080)
        streammux.set_property('batched-push-timeout', MUXER_BATCH_TIMEOUT_USEC)
    streammux.set_property('batch-size', 1)
    pgie.set_property('config-file-path', "peoplenet_infer_config.txt")

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(demux)
    pipeline.add(h264parser)
    pipeline.add(decoder)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    source.link(demux)
    
    # Connect qtdemux pad-added signal for dynamic linking
    demux.connect("pad-added", cb_newpad, (pipeline, h264parser))

    # Link remaining elements
    h264parser.link(decoder)
    sinkpad = streammux.request_pad_simple("sink_0")
    if not sinkpad:
        sys.stderr.write("Unable to get the sink pad of streammux \n")
        sys.exit(1)
    srcpad = decoder.get_static_pad("src")
    if not srcpad:
        sys.stderr.write("Unable to get source pad of decoder \n")
        sys.exit(1)
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get sink pad of nvosd \n")
        sys.exit(1)
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))