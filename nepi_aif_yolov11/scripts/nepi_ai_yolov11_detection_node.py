#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

import os
import time
import copy
import sys
import cv2
import numpy as np


from nepi_sdk import nepi_sdk
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_img
from nepi_sdk import nepi_ais

from nepi_api.ai_if_detector import AiDetectorIF
from nepi_api.messages_if import MsgIF

# Define your PyTorch model and load the weights
# model = ...


class Yolov11Detector():
    default_config_dict = {'threshold': 0.3,'max_rate': 5}
    #######################
    ### Node Initialization
    DEFAULT_NODE_NAME = "ai_yolov11" # Can be overwitten by luanch command
    MODEL_FRAMEWORK="yolov11"

    def __init__(self):
        ####  NODE Initialization ####
        nepi_sdk.init_node(name= self.DEFAULT_NODE_NAME)
        self.class_name = type(self).__name__
        self.base_namespace = nepi_sdk.get_base_namespace()
        self.node_name = nepi_sdk.get_node_name()
        self.node_namespace = nepi_sdk.get_node_namespace()

        ##############################  
        # Create Msg Class
        self.msg_if = MsgIF(log_name = self.class_name)
        self.msg_if.pub_info("Starting Node Initialization Processes")

       ##############################  
        # Initialize Class Variables

        ############  Get ALL_NAMESPACE if provided
        param_namespace = nepi_sdk.create_namespace(self.node_namespace,'all_namespace')
        self.all_namespace = nepi_sdk.get_param(param_namespace,"")
        if self.all_namespace == "":
            self.all_namespace = self.node_namespace


        ############  Get WEIGHT_FILE Path
        param_namespace = nepi_sdk.create_namespace(self.node_namespace,'weight_file_path')
        self.weight_file_path = str(nepi_sdk.get_param(param_namespace,""))
        self.msg_if.pub_warn("Got weight file path: " + self.weight_file_path)
        if self.weight_file_path == "" or os.path.exists(self.weight_file_path) == False:
            self.msg_if.pub_warn("Failed to get required node info from param server at: " + str(param_namespace))
            nepi_sdk.signal_shutdown("Failed to get valid weight path, got: " + self.weight_file_path)
            return

        ############  Get PARAMS_FILE Path
        param_namespace = nepi_sdk.create_namespace(self.node_namespace,'param_file_path')
        self.param_file_path = str(nepi_sdk.get_param(param_namespace,""))
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
                self.msg_if.pub_warn("Model not a " + self.MODEL_FRAMEWORK  + " model: " + model_framework)
                nepi_sdk.signal_shutdown("Model not a valid framework")
                return

            if model_type != 'detection':
                self.msg_if.pub_warn("Model not a valid type: " + model_type)
                nepi_sdk.signal_shutdown("Model not a valid type")
                return
            


            ##############################  
            # Load Model

            # Import ultralytics here so we can message
            self.msg_if.pub_warn("Importing torch and ultralytics YOLO package")
            import torch
            from ultralytics import YOLO


            self.device = 'cpu'
            has_cuda = torch.cuda.is_available()
            self.msg_if.pub_warn("CUDA available: " + str(has_cuda))
            if has_cuda == True:
                cuda_count = torch.cuda.device_count()
                self.msg_if.pub_warn("CUDA GPU Count: " + str(cuda_count))
                if cuda_count > 0:
                    self.device = 'cuda'

            ##############################  


            # Initialize Detector with Blank Img
            self.msg_if.pub_warn("Initializing detector with blank img")
            init_cv2_img=nepi_img.create_cv2_blank_img()
            det_dict=self.processDetection(init_cv2_img)

            # Run Tests
            NUM_TESTS=10
            self.msg_if.pub_warn("Running Detection Speed Test on " + str(NUM_TESTS) + " Images")
            start_time = time.time()
            for i in range(1, NUM_TESTS):
                det_dict=self.processDetection(init_cv2_img)
            elapsed_time = round( ( time.time() - start_time ) , 3)  # Slower for real images
            detect_time = round( elapsed_time / NUM_TESTS , 2)
            detect_rate = round( float(1.0)/detect_time , 2)
            self.msg_if.pub_warn("Average Detection Time: " + str(detect_time) + " sec")
            self.msg_if.pub_warn("Average Detection Rate: " + str(detect_rate) + " hz")

            # Create API IF Class
            self.msg_if.pub_info("Starting ai_if with default_config_dict: " + str(self.default_config_dict))
            self.ai_if = AiDetectorIF(
                                namespace = self.node_namespace,
                                model_name = self.node_name,
                                framework = model_framework,
                                description = model_description,
                                proc_img_height = self.proc_img_height,
                                proc_img_width = self.proc_img_width,
                                classes_list = self.classes,
                                default_config_dict = self.default_config_dict,
                                all_namespace = self.all_namespace,
                                processDetectionFunction = self.processDetection,
                                has_img_tiling = False)

            #########################################################
            ## Initiation Complete
            
   

            # Spin forever (until object is detected)
            nepi_sdk.spin()
            #########################################################        
              




    def processDetection(self, cv2_img, img_dict=dict(), threshold = 0.3, resize = False, verbose = False):

        img_dict['image_width'] = 1
        img_dict['image_height'] = 1 
        img_dict['prc_width'] = 1
        img_dict['prc_height'] = 1 
        img_dict['ratio'] = 1
        img_dict['tiling'] = False

        detect_dict_list = []
        if cv2_img is not None:

                cv2_img_shape = cv2_img.shape
                cv2_img_width = cv2_img_shape[1]
                cv2_img_height = cv2_img_shape[0]
                cv2_img_area = cv2_img_shape[0] * cv2_img_shape[1]

                if resize == True:
                    [cv2_img,rescale_ratio,prc_width,prc_height] = nepi_img.resize_proportionally(cv2_img, self.proc_img_width,self.proc_img_height,interp = cv2.INTER_NEAREST)
                else:
                    rescale_ratio = 1
                    prc_width = cv2_img_width
                    prc_height = cv2_img_height

                # Convert to RGB
                if nepi_img.is_gray(cv2_img):
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2BGR)
                else:
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

                #self.msg_if.pub_info(":yolov5: Preprocessed image with image size: " + str(cv2_img.shape))
                # Create image dict with new image
                img_dict['image_width'] = cv2_img_width 
                img_dict['image_height'] = cv2_img_height 
                img_dict['prc_width'] = prc_width 
                img_dict['prc_height'] = prc_height 
                img_dict['ratio'] = rescale_ratio 
                img_dict['tiling'] = False


                # Update model settings
                self.model.conf = threshold  # Confidence threshold (0-1)

                try:
                    # Inference
                    start_time = nepi_sdk.get_time()

                    results = self.model(cv2_img, conf=threshold, verbose=False) #, device=self.device)

                    detect_time = round( (nepi_sdk.get_time() - start_time) , 3)


                    ids = results[0].boxes.cls.to('cpu').tolist()
                    boxes = results[0].boxes.xyxy.to('cpu').tolist()
                    confs = results[0].boxes.conf.to('cpu').tolist()

                    # self.msg_if.pub_warn("Got detection ids: " + str(ids))
                    # self.msg_if.pub_warn("Got detection boxes: " + str(boxes))
                    # self.msg_if.pub_warn("Got detection confs: " + str(confs))
                
                except Exception as e:
                    self.msg_if.pub_info("Failed to process detection with exception: " + str(e))
            
                
                for i, idf in enumerate(ids):
                    id = int(idf)
                    det_name = self.classes[id]
                    det_id = id
                    det_prob = confs[i]
                    det_box = boxes[i]
                    det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
                    detect_dict = {
                        'name': det_name, # Class String Name
                        'id': det_id, # Class Index from Classes List
                        'uid': '', # Reserved for unique tracking by downstream applications
                        'prob': det_prob, # Probability of detection
                        'xmin': int(det_box[0] ),
                        'ymin': int(det_box[1] ) ,
                        'xmax': int(det_box[2] ),
                        'ymax': int(det_box[3]),
                        'area_pixels': int(det_area),
                        'area_ratio': det_area / cv2_img_area
                    }
                    # Rescale to orig image size
                    detect_dict['xmin'] = int(detect_dict['xmin'] * rescale_ratio)
                    detect_dict['ymin'] = int(detect_dict['ymin'] * rescale_ratio)
                    detect_dict['xmax'] = int(detect_dict['xmax'] * rescale_ratio)
                    detect_dict['ymax'] = int(detect_dict['ymax'] * rescale_ratio)
                    detect_dict_list.append(detect_dict)


                    if verbose == True:
                        self.msg_if.pub_info("Detector Detect Time: " + str(detect_time))
                        self.msg_if.pub_info("Got detect dict entry: " + str(detect_dict))
            
        return [detect_dict_list, img_dict]





if __name__ == '__main__':
    Yolov11Detector()
