<!--
Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.

This file is part of nepi-engine
(see https://github.com/nepi-engine).

License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
-->
# nepi_ai_frameworks #
This repository provides the _nepi_ai_frameworks_ ROS support files. The _nepi_ai_frameworks_ run in NEPI’s AI Model Management (AIM) system managed by the NEPI ai_detector_mgr Node, which provides a framework for storing, connecting, and running AI models from an onboard AI model library, as well as publishing AIM results to downstream consumers like a NEPI automation script of connected control system. The AIM system manages which of the available AI models is active at a given time, which sensor stream is provided as input to the active model, and any additional parameters related to AI model execution. The AIM system also supports external AI process orchestration such as active model selection and on-the-fly data input switching. The AIM system supports AI model output logging and ensures AI model outputs adhere to a standard ROS 2D label topic format, ensuring a consistent interface for both on-board and off-board consumers of the AIM data output streams.

On bootup, the AIM system queries all the models stored in the AIM Onboard AI Model Library located on the user storage drive mounted in the NEPI File System at:

	/mnt/nepi_storage/ai_models

This folder is also accessible and editable from a PC’s File Manager application connected to the NEPI user storage drive at:

	smb://192.168.179.103/nepi_storage/ai_models

NOTE: Models deployed into the AIM’s AI Model Library must be supported by one of the AIM AI frameworks installed on your NEPI device and placed into the appropriate library subfolders for the specific AI framework you intend to run your models with. To see a list of AIM supported AI frameworks, required files, and AIM Model Library deployment folders, see the document “NEPI Engine – Supported AI Frameworks and Models” available at:

	https://nepi.com/documentation/

### Build and Install ###
This repository is typically built as part of the [nepi_engine_ws](https://github.com/nepi-engine/nepi_engine_ws) catkin workspace. Refer to that project's top-level README for build and install instructions.

The repository may be included in other custom catkin workspaces or built using standard CMake commands (with appropriate variable definitions provided), though it should be noted that the executables here work in concert with a complete NEPI Engine installation, so operability and functionality may be limited when building outside of the [nepi_engine_ws](https://github.com/nepi-engine/nepi_engine_ws) source tree.

### Branching and Tagging Strategy ###
In general branching, merging, tagging are all limited to the [nepi_engine_ws](https://github.com/nepi-engine/nepi_engine_ws) container repository, with the present submodule repository kept as simple and linear as possible.

### Contribution guidelines ###
Bug reports, feature requests, and source code contributions (in the form of pull requests) are gladly accepted!

### Who do I talk to? ###
At present, all user contributions and requests are vetted by Numurus technical staff, so you can use any convenient mechanism to contact us.
