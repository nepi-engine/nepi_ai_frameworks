#[[
Copyright (c) 2024 Numurus, LLC <https://www.numurus.com>.

This file is part of nepi-engine
(see https://github.com/nepi-engine).

License: 3-clause BSD, see https://opensource.org/licenses/BSD-3-Clause
]]
cmake_minimum_required(VERSION 3.5)
project(nepi_aif_hailo)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
std_msgs
)

catkin_package(
  CATKIN_DEPENDS std_msgs
)



###########
## Build ##
###########



#############
## Install ##
#############


if(NOT NEPI_BASE)
  set( NEPI_BASE
    /opt/nepi
  )
endif()

if(NOT NEPI_RUI)
  set( NEPI_RUI
    ${NEPI_BASE}/nepi_rui
  )
endif()

if(NOT NEPI_ENGINE)
  set( NEPI_ENGINE
    ${NEPI_BASE}/nepi_engine
  )
endif()

if(NOT NEPI_ETC)
  set( NEPI_ETC
    ${NEPI_BASE}/etc
  )
endif()


catkin_install_python(PROGRAMS
  scripts/nepi_ai_hailo_detection_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


set( NEPI_AIF_SHARE_FOLDER
  ${NEPI_ENGINE}/share/nepi_aifs
)

install(DIRECTORY params/
  DESTINATION ${NEPI_AIF_SHARE_FOLDER}
)

set( NEPI_API_FOLDER
  ${NEPI_ENGINE}/lib/python3/dist-packages/nepi_api
)

install(DIRECTORY api/
  DESTINATION ${NEPI_API_FOLDER}
  FILES_MATCHING PATTERN "*.py"
)


#############
## Testing ##
#############
