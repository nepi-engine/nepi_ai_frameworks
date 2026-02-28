#!/usr/bin/env python
#
# Copyright (c) 2024 Numurus <https://www.numurus.com>.
#
# License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
#

import os
import os.path



from nepi_sdk import nepi_sdk
from nepi_sdk.nepi_sdk import logger as Logger
from nepi_sdk import nepi_aifs



from std_msgs.msg import Empty, Float32, Int32, String, Bool


TEST_AI_DICT = {
'description': 'Yolov11 ai framework support', 
'pkg_name': 'nepi_aif_yolov11', 
'if_file_name': 'aif_yolov11_if.py', 
'if_path_name': '/opt/nepi/nepi_engine/share/nepi_aifs', 
'if_module_name': 'aif_yolov11_if', 
'if_class_name': 'Yolov11AIF', 
'models_folder_name': 'yolov11', 
'launch_pkg_name': 'nepi_aif_yolov11',
'launch_file_name': 'yolov11_ros.launch', 
'node_file_name': 'nepi_ai_yolov11_detection_node.py',  
'active': True
}

TEST_LAUNCH_NAMESPACE = "/nepi/yolov11_test"
TEST_MGR_NAMESPACE = "/nepi/ai_detector_mgr"
TEST_MODELS_LIB_PATH = "/mnt/nepi_storage/ai_models/"


MODEL_FRAMEWORK="yolov11"

class Yolov11AIF(object):

    node_dict = dict()

    def __init__(self, aif_dict, launch_namespace, models_lib_path):
      self.pkg_name = aif_dict['pkg_name']
      self.log_name = self.pkg_name
      self.logger = Logger(log_name = self.log_name) 
      self.logger.log_warn("Instantiating with aif_dict: " +str(aif_dict))

      if launch_namespace[-1] == "/":
        launch_namespace = launch_namespace[:-1]
      self.launch_namespace = launch_namespace  
      #self.logger.log_warn("Launch Namespace: " + self.launch_namespace)
      self.models_lib_path = models_lib_path

      self.node_file_dict = aif_dict['node_file_dict']
      self.models_folder = aif_dict['models_folder_name']
      self.models_folder_path =  os.path.join(self.models_lib_path, self.models_folder)
    
    #################
    # Model Functions

    def getModelsDict(self):
        # Try to obtain the path to MODEL_FRAMEWORK models from the system_mgr
        self.logger.log_warn("Looking for model files in folder: " + self.models_folder_path)
        # Grab the list of all existing cfg files
        models_dict = nepi_aifs.loadModelsDict(MODEL_FRAMEWORK, self.pkg_name, self.models_folder_path)
        ##################
        # Add custom entries to models_dict if needed here.
        ##################
        self.logger.log_warn("Returning models dict" + str(models_dict.keys()))
        return models_dict


    def launchModel(self, model_dict):
        #self.logger.log_warn("Launching Model Node with model dict" + str(model_dict))
        [success, node_namespace, self.node_dict] = nepi_aifs.launchModelNode(model_dict,self.node_file_dict,self.launch_namespace,self.node_dict)
        return success, node_namespace



    def killModel(self,model_name):
        [success, self.node_dict] = nepi_aifs.killModelNode(model_name,self.node_dict)
        return success
 
   

if __name__ == '_main_':
    node_name = "ai_model_test"
    while nepi_sdk.check_for_node(node_name):
        nepi_sdk.kill_node(node_name)
        nepi_sdk.sleep(2,10)
    Yolov8AIF(TEST_AI_DICT,TEST_LAUNCH_NAMESPACE,TEST_MGR_NAMESPACE,TEST_MODELS_LIB_PATH)
