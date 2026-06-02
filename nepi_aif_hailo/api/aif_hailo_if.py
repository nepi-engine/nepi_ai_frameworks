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
import os.path


from nepi_sdk import nepi_sdk
from nepi_sdk import nepi_utils
from nepi_sdk import nepi_aifs
from nepi_sdk.nepi_sdk import logger as Logger

from std_msgs.msg import Empty, Float32, Int32, String, Bool


TEST_AI_DICT = {
'description': 'Hailo ai framework support',
'pkg_name': 'nepi_aif_hailo',
'if_file_name': 'aif_hailo_if.py',
'if_path_name': '/opt/nepi/nepi_engine/share/nepi_aifs',
'if_module_name': 'aif_hailo_if',
'if_class_name': 'HailoAIF',
'models_folder_name': 'hailo',
'node_file_name': 'nepi_ai_hailo_detection_node.py',
'active': True
}

TEST_LAUNCH_NAMESPACE = "/nepi/hailo_test"
TEST_MGR_NAMESPACE = "/nepi/ai_detector_mgr"
TEST_MODELS_LIB_PATH = "/mnt/nepi_storage/ai_models/"

MODEL_FRAMEWORK = "hailo"


class HailoAIF(object):

    node_dict = dict()

    def __init__(self, aif_dict, launch_namespace, models_lib_path):
        self.pkg_name = aif_dict['pkg_name']
        self.log_name = self.pkg_name
        self.logger = Logger(log_name=self.log_name)
        self.logger.log_warn("Instantiating with aif_dict: " + str(aif_dict))

        if launch_namespace[-1] == "/":
            launch_namespace = launch_namespace[:-1]
        self.launch_namespace = launch_namespace
        self.models_lib_path = models_lib_path

        self.node_file_dict = aif_dict['node_file_dict']
        self.models_folder = aif_dict['models_folder_name']
        self.models_folder_path = os.path.join(self.models_lib_path, self.models_folder)


    #################
    # Framework Functions

    def checkFrameworkSupport(self):
        supported = True

        if supported == True:
            check = 'cv2'
            if nepi_utils.check_module_available(check) == False:
                supported = False
                self.logger.log_warn("Framework failed check: " + check)

        if supported == True:
            check = 'is_valid_hailo'
            if nepi_utils.bash_nepi_check(check) == False:
                supported = False
                self.logger.log_warn("Framework failed check: " + check)
                
        return supported

    #################
    # Model Functions

    def getModelsDict(self):
        WEIGHTS_FOLDER = None
        hailo_hw_version = str(nepi_utils.bash_nepi_get('get_hailo_hw_version')).replace(" ","")
        if hailo_hw_version == "8":
            self.logger.log_warn("Got Hailo HW Version: " + hailo_hw_version)
            WEIGHTS_FOLDER = 'hailo8'
        elif hailo_hw_version == "10":
            self.logger.log_warn("Got Hailo HW Version: " + hailo_hw_version)
            WEIGHTS_FOLDER = 'hailo10'
        else:
            self.logger.log_warn("Got Unknown Hailo HW Version: " + hailo_hw_version)


        if WEIGHTS_FOLDER is not None:
            models_folder_path = os.path.join(self.models_folder_path, WEIGHTS_FOLDER)




        self.logger.log_warn("Looking for model files in folder: " + models_folder_path)
        models_dict = nepi_aifs.loadModelsDict(MODEL_FRAMEWORK, self.pkg_name, models_folder_path)
        ##################
        # Add custom entries to models_dict if needed here.
        ##################
        self.logger.log_warn("Returning models dict" + str(models_dict.keys()))
        self.logger.log_warn("Returning models dict" + str(models_dict))
        return models_dict

    def launchModel(self, model_dict):
        self.logger.log_warn("Launching Model Node with model dict" + str(model_dict))
        [success, node_namespace, self.node_dict] = nepi_aifs.launchModelNode(model_dict, self.node_file_dict, self.launch_namespace, self.node_dict)
        return success, node_namespace

    def killModel(self, node_name):
        self.logger.log_warn("Killing Node " + str(node_name))
        [success, self.node_dict] = nepi_aifs.killModelNode(node_name, self.node_dict)
        return success


if __name__ == '_main_':
    node_name = "ai_model_test"
    while nepi_sdk.check_for_node(node_name):
        nepi_sdk.kill_node(node_name)
        nepi_sdk.sleep(2, 10)
    HailoAIF(TEST_AI_DICT, TEST_LAUNCH_NAMESPACE, TEST_MGR_NAMESPACE, TEST_MODELS_LIB_PATH)
