#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# This file is part of nepi applications (nepi_ai_frameworks) repo
# (see https://https://github.com/nepi-engine/nepi_ai_frameworks)
#
# License: nepi applications are licensed under the "Numurus Software License",
# which can be found at: <https://numurus.com/wp-content/uploads/Numurus-Software-License-Terms.pdf>
#
# Redistributions in source code must retain this top-level comment bstab.
# Plagiarizing this software to sidestep the license obligations is illegal.
#
# Contact Information:
# ====================
# - mailto:nepi@numurus.com

import os
import time
import copy
import sys
import cv2
import numpy as np
from PIL import Image


try:
    from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType)
    from hailo_platform import *
    HAILO_AVAILABLE = True
    HAILO_IMPORT_ERROR = None
except ImportError as e:
    print("FAILED TO LOAD hailo_platform")
    HEF = None
    HAILO_AVAILABLE = False
    HAILO_IMPORT_ERROR = str(e)

HAILO_INTERFACE = HailoStreamInterface.PCIe

from nepi_sdk import nepi_sdk
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img
from nepi_sdk import nepi_ais

from nepi_api.process_if_ai_detector import AiDetectorIF
from nepi_api.messages_if import MsgIF



class HailoDetector():
    default_config_dict = {'threshold': 0.3, 'max_rate': 5}

    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_hailo"
    MODEL_FRAMEWORK = "hailo"

    def __init__(self):
        ####  NODE Initialization ####
        nepi_sdk.init_node(name=self.DEFAULT_NODE_NAME)
        self.class_name = type(self).__name__
        self.base_namespace = nepi_sdk.get_base_namespace()
        self.node_name = nepi_sdk.get_node_name()
        self.node_namespace = nepi_sdk.get_node_namespace()

        ##############################
        # Create Msg Class
        self.msg_if = MsgIF(log_name=self.class_name)
        self.msg_if.pub_info("Starting Node Initialization Processes")

        ##############################
        # Initialize Class Variables

        ############  Get ALL_NAMESPACE if provided
        param_namespace = nepi_sdk.create_namespace(self.node_namespace, 'all_namespace')
        self.all_namespace = nepi_sdk.get_param(param_namespace, "")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace

        ############  Get WEIGHT_FILE Path
        param_namespace = nepi_sdk.create_namespace(self.node_namespace, 'weight_file_path')
        self.weight_file_path = str(nepi_sdk.get_param(param_namespace, ""))
        self.msg_if.pub_warn("Got weight file path: " + self.weight_file_path)
        if self.weight_file_path == "" or os.path.exists(self.weight_file_path) == False:
            self.msg_if.pub_warn("Failed to get required node info from param server at: " + str(param_namespace))
            nepi_sdk.signal_shutdown("Failed to get valid weight path, got: " + self.weight_file_path)
            return

        ############  Get PARAMS_FILE Path
        param_namespace = nepi_sdk.create_namespace(self.node_namespace, 'param_file_path')
        self.param_file_path = str(nepi_sdk.get_param(param_namespace, ""))
        self.msg_if.pub_warn("Got param file path: " + self.param_file_path)
        if self.param_file_path == "" or os.path.exists(self.param_file_path) == False:
            self.msg_if.pub_warn("Failed to get required node info from param server at: " + str(param_namespace))
            nepi_sdk.signal_shutdown("Failed to get valid param path, got: " + self.param_file_path)
            return

        ############### Load Model Params
        yaml_dict = nepi_utils.read_dict_from_file(self.param_file_path)

        self.msg_if.pub_warn("Got model info: " + str(yaml_dict))

        if yaml_dict is None:
            self.msg_if.pub_warn("Failed load model info dict from: " + str(self.param_file_path))
            nepi_sdk.signal_shutdown("Failed to get valid model info from param: " + str(self.param_file_path))
            return
        else:
            try:
                model_info_dict = yaml_dict['ai_model']
                model_framework = model_info_dict['framework']['name']
                model_type = model_info_dict['type']['name']
                model_description = model_info_dict['description']['name']
                self.classes = model_info_dict['classes']['names']
                self.proc_img_width = model_info_dict['image_size']['image_width']['value']
                self.proc_img_height = model_info_dict['image_size']['image_height']['value']
            except Exception as e:
                self.msg_if.pub_warn("Failed to get required model info from params: " + str(e))
                nepi_sdk.signal_shutdown("Failed to get valid model file paths")
                return

            if model_framework != self.MODEL_FRAMEWORK:
                self.msg_if.pub_warn("Model not a " + self.MODEL_FRAMEWORK + " model: " + model_framework)
                nepi_sdk.signal_shutdown("Model not a valid framework")
                return

            if model_type != 'detection':
                self.msg_if.pub_warn("Model not a valid type: " + model_type)
                nepi_sdk.signal_shutdown("Model not a valid type")
                return

            ##############################
            # Load Model

            # self.msg_if.pub_warn("Importing hailo_platform package")
            # from hailo_platform import HEF, VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

            if HEF is None:
                "Failed to load hailo_platfrom module"
            else:

                self.device = VDevice()
                self.msg_if.pub_warn("Loading HEF model: " + self.weight_file_path)
                self.hef = HEF(self.weight_file_path)      
                # Configure the network group
                self.configure_params = None
                self.network_groups = None
                self.network_group = None
                self.network_group_params = None
                try:
                    self.configure_params = ConfigureParams.create_from_hef(self.hef, interface=HAILO_INTERFACE)
                    self.network_group = self.device.configure(self.hef, self.configure_params)[0]
                    self.network_group_params = self.network_group.create_params()
                except Exception as e:
                    print("Device config failed with error: " + str(e))
                if self.configure_params is not None and self.network_group_params is not None:
                    print("Got network config: " + str(self.network_group_params))
                    # Get stream info for input/output naming
                    self.input_vstream_info = self.hef.get_input_vstream_infos()[0]
                    self.input_vstreams_params = InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.UINT8)
                    print("")
                    print("self.input_vstream_info " + str(self.input_vstreams_params))
                    
                    self.output_vstreams_params = OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=FormatType.FLOAT32)
                    print("")
                    print("self.output_vstreams_params " + str(self.output_vstreams_params))
                    print("")
                    # Build vstream params

                # Initialize with blank image
                self.msg_if.pub_warn("Initializing detector with blank img")
                init_cv2_img = nepi_img.create_cv2_blank_img()
                det_dict = self.processImage(init_cv2_img)

                # Speed test
                NUM_TESTS = 10
                self.msg_if.pub_warn("Running Detection Speed Test on " + str(NUM_TESTS) + " Images")
                start_time = time.time()
                for i in range(1, NUM_TESTS):
                    det_dict = self.processImage(init_cv2_img)
                elapsed_time = round((time.time() - start_time), 4)
                detect_time = round(elapsed_time / NUM_TESTS, 4) + 0.0001
                detect_rate = round(float(1.0) / detect_time, 4)
                self.msg_if.pub_warn("Average Detection Time: " + str(detect_time) + " sec")
                self.msg_if.pub_warn("Average Detection Rate: " + str(detect_rate) + " hz")

                # Create API IF Class
                self.msg_if.pub_info("Starting ai_if with default_config_dict: " + str(self.default_config_dict))
                self.ai_if = AiDetectorIF(
                    namespace=self.node_namespace,
                    model_name=self.node_name,
                    framework=model_framework,
                    description=model_description,
                    proc_img_height=self.proc_img_height,
                    proc_img_width=self.proc_img_width,
                    classes_list=self.classes,
                    default_config_dict=self.default_config_dict,
                    all_namespace=self.all_namespace,
                    processImageFunction=self.processImage,
                    processFileFunction=self.processFile,
                    has_img_tiling=False)

                nepi_sdk.spin()


    def processImage(self, cv2_img, img_dict=dict(), threshold=0.3, resize=True, verbose=False):

        img_dict['image_width'] = 0
        img_dict['image_height'] = 0
        img_dict['prc_width'] = 0
        img_dict['prc_height'] = 0
        img_dict['ratio'] = 1
        img_dict['tiling'] = False

        detect_dict_list = []
        if cv2_img is not None:

            if nepi_img.is_gray(cv2_img):
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

    

            # Get expected model input dimensions
            input_info = self.hef.get_input_stream_infos()[0]
            input_shape = input_info.shape
            #print(input_shape)
            input_height, input_width = input_shape[0], input_shape[1]

            # Resize image to match model input and normalize (if required by your pre-processing)
            cv2_img_shape = cv2_img.shape
            cv2_img_width = cv2_img_shape[1]
            cv2_img_height = cv2_img_shape[0]
            cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]



            resized_image = cv2.resize(img_rgb, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
            
            img_dict['image_width'] = cv2_img_width
            img_dict['image_height'] = cv2_img_height
            img_dict['prc_width'] = input_width
            img_dict['prc_height'] = input_height
            img_dict['ratio'] = 1
            img_dict['tiling'] = False

            # Normalize the image (typically [0, 255] -> [0, 1] or [-1, 1] depending on your HEF model)
            #input_data = resized_image.astype(np.float32) / 255.0 

            input_data = {self.input_vstream_info.name: np.expand_dims(resized_image, axis=0).astype(np.uint8)} 
            if input_data is not None:
                    

                    #######################
                    start_time = nepi_sdk.get_time()

                    results = None
                    try:
                        with InferVStreams(self.network_group, self.input_vstreams_params, self.output_vstreams_params) as infer_pipeline:   
                                    with self.network_group.activate(self.network_group_params):
                                        results = infer_pipeline.infer(input_data)
                       
                    except Exception as e:
                        print("Failed to process detection with exception: " + str(e))

                    detect_time = round((nepi_sdk.get_time() - start_time), 3)


                    #######################
                    if results is not None:
                        try:
                            for class_ind, class_detections in enumerate(results[list(results.keys())[0]][0]):
                                if class_detections.shape[0]>0:

                                    for detection in class_detections:
                                        if len(detection) >= 5:
                                            if detection[4] > threshold:
                                                
                                                #print(detection)
                                                det_name = self.classes[class_ind]
                                                det_id = class_ind
                                                det_prob = round(detection[4].item(), 5)
                                                [ymin, xmin, ymax, xmax] = detection[:4]

                                                # Convert to pixel coordinates relative to the original image
                                                oh, ow, _ = np.asarray(cv2_img).shape
                                                rh, rw = oh/input_shape[0], ow/input_shape[1]
                                                abs_ymin = int(ymin * oh)
                                                abs_xmin = int(xmin * ow)
                                                abs_ymax = int(ymax * oh)
                                                abs_xmax = int(xmax * ow)

                                                det_area = (abs_xmax - abs_xmin) * (abs_ymax - abs_ymin)
                                                detect_dict = {
                                                    'name': det_name,
                                                    'id': det_id,
                                                    'uid': '',
                                                    'prob': det_prob,
                                                    'xmin': abs_xmin,
                                                    'ymin': abs_ymin,
                                                    'xmax': abs_xmax,
                                                    'ymax': abs_ymax,
                                                    'area_pixels': int(det_area),
                                                    'area_ratio': det_area / cv2_img_area
                                                }
                                                detect_dict_list.append(detect_dict)
                        except Exception as e:
                            self.msg_if.pub_info("Failed to process detection with exception: " + str(e))
        # if len(detect_dict_list) > 0:
        #     print(detect_dict_list[0])
        return [detect_dict_list, img_dict]


    def processFile(self, img_file, img_dict=dict(), threshold=0.3, resize=False, verbose=False):

        img_dict['image_width'] = 1
        img_dict['image_height'] = 1
        img_dict['prc_width'] = 1
        img_dict['prc_height'] = 1
        img_dict['ratio'] = 1
        img_dict['tiling'] = False

        detect_dict_list = []
        if img_file is not None:
            if os.path.exists(img_file) == True:
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                except:
                    if verbose == True:
                        self.msg_if.pub_info("Failed to read meta data from image file: " + str(img_file))
                    [width, height] = [None, None]

                if width is not None and height is not None:
                    cv2_img = cv2.imread(img_file)
                    if cv2_img is not None:
                        [detect_dict_list, img_dict] = self.processImage(
                            cv2_img, img_dict=img_dict, threshold=threshold, resize=resize, verbose=verbose)

        return [detect_dict_list, img_dict]



if __name__ == '__main__':
    HailoDetector()
