# nepi_ai_frameworks — Developer Reference

## Purpose

`nepi_ai_frameworks` provides pluggable AI model framework adapters for the NEPI platform. Each adapter is an independent ROS package that implements the NEPI AI Framework Interface (AIF) for a specific inference framework (currently YOLOv8 and YOLOv11). The `ai_models_mgr` node in `nepi_engine` discovers and invokes these adapters to enumerate available models and launch inference nodes. This submodule also contains training infrastructure for producing new YOLO-compatible models.

## Architecture

```
nepi_ai_frameworks/
├── nepi_aif_yolov8/           # YOLOv8 framework adapter
│   ├── api/
│   │   └── aif_yolov8_if.py   # Yolov8AIF class: model enumeration and node launch
│   ├── scripts/
│   │   └── nepi_ai_yolov8_detection_node.py   # ROS inference node
│   ├── params/
│   │   └── aif_yolov8_params.yaml             # Framework config
│   ├── CMakeLists.txt
│   └── package.xml
│
├── nepi_aif_yolov11/          # YOLOv11 framework adapter
│   ├── api/
│   │   └── aif_yolov11_if.py  # Yolov11AIF class: model enumeration and node launch
│   ├── scripts/
│   │   └── nepi_ai_yolov11_detection_node.py  # ROS inference node
│   ├── params/
│   │   └── aif_yolov11_params.yaml            # Framework config
│   ├── CMakeLists.txt
│   └── package.xml
│
└── nepi_ai_training/          # Model training utilities
    ├── nepi_yolo_detector_training/
    │   ├── initialize_project_yolo_detector.py
    │   ├── label_data_yolo_detector.py
    │   ├── train_model_yolo_detector.py
    │   ├── deploy_model_yolo_detector.py
    │   └── yolo_detector_utils.py
    ├── TRAIN_CUSTOM_YOLO_AI_MODEL.md
    ├── ai_train_env_setup.sh
    ├── ai_train_yolo_project_setup.sh
    ├── CMakeLists.txt
    ├── setup.py
    └── package.xml
```

`nepi_ai_frameworks` itself has a `.gitmodules` — the two framework adapters and the training package are organized as nested submodules of this submodule.

## How It Works

**Framework adapter pattern:**

Each adapter package implements a class (e.g., `Yolov8AIF`, `Yolov11AIF`) with two responsibilities:
1. `getModelsDict()` — scans the model storage directory (e.g., `/mnt/nepi_storage/ai_models/yolov8/`) for available `.pt` or similar model files and returns a dictionary describing each model
2. `launchModelNode(model_dict)` — calls `nepi_aifs.launchModelNode()` from the `nepi_sdk` to spawn the framework-specific inference ROS node for a selected model

`ai_models_mgr` in `nepi_engine` calls `getModelsDict()` periodically to keep the available model list current, and calls `launchModelNode()` when the user activates a model via the RUI.

**Inference node pattern:**

Each `nepi_ai_*_detection_node.py` is a standalone ROS node. It:
1. Loads a YOLO model from the path passed via `~drv_dict` or equivalent param
2. Subscribes to a configured image topic
3. Runs inference on received frames
4. Publishes detection results as `AiBoundingBoxes` (from `nepi_interfaces`)

The detection output format is consistent across frameworks — downstream nodes (apps, automation scripts) do not need to know which framework produced the detections.

**Training workflow (`nepi_ai_training`):**

A four-stage pipeline for producing custom models:
1. `initialize_project_yolo_detector.py` — sets up project directory structure
2. `label_data_yolo_detector.py` — assists with YOLO-format data labeling
3. `train_model_yolo_detector.py` — invokes Ultralytics training
4. `deploy_model_yolo_detector.py` — copies trained model to the NEPI model storage directory

Training runs off-device (development machine with GPU) using the Ultralytics YOLO Python package. The deployed model is transferred to `/mnt/nepi_storage/ai_models/{framework}/` on the NEPI device.

## ROS Interface

**Published by inference nodes:**
- `{namespace}/bounding_boxes` (`AiBoundingBoxes`) — detection results per frame
- `{namespace}/status` (`AiDetectorStatus`) — inference node status

**Subscribed by inference nodes:**
- A configured image topic (`sensor_msgs/Image`) — source frames for inference

Exact topic names are set by the framework adapter when launching the node and follow the NEPI device namespace convention.

## Build and Dependencies

Each adapter is a catkin package built as part of the workspace. No standalone build.

Runtime dependencies (must be installed on the NEPI device):
- `nepi_sdk`, `nepi_api`, `nepi_interfaces` — NEPI platform base
- `rospy`, `std_msgs`, `sensor_msgs` — ROS base
- Ultralytics YOLO Python package — `pip install ultralytics` (provides both v8 and v11 via version selection)
- GPU support optional but expected for real-time inference; CPU-only inference is possible but slow

Model files live at:
- `/mnt/nepi_storage/ai_models/yolov8/` — YOLOv8 models
- `/mnt/nepi_storage/ai_models/yolov11/` — YOLOv11 models

These paths are read by the AIF classes and must exist on the target device. The `ai_models_mgr` creates these directories if they do not exist.

Training dependencies (development machine only):
- `ultralytics` (Ultralytics YOLO package)
- `torch`, `torchvision` — PyTorch
- GPU with CUDA recommended for practical training times

## Known Constraints and Fragile Areas

**Model storage paths are hardcoded per framework.** Each AIF class looks in a fixed subdirectory under `/mnt/nepi_storage/ai_models/`. Adding a new framework requires creating its own storage directory and registering it with `ai_models_mgr`.

**No version pinning on Ultralytics.** The `ultralytics` package is not pinned to a specific version in any `requirements.txt` visible in this submodule. Ultralytics releases have historically changed model loading APIs between minor versions. A `pip install ultralytics` may install a version incompatible with the detection node code.

**Nested submodule structure.** `nepi_ai_frameworks` contains its own `.gitmodules` — the adapters and training package are submodules within this submodule. When cloning the workspace, `git submodule update --init --recursive` is required to fully populate this directory. A non-recursive submodule update will leave the adapter directories empty.

**Only detection models are supported.** The current AIF interface and `ai_models_mgr` are designed for object detection models (`MODEL_TYPE_LIST = ['detection']`). Classification, segmentation, and pose estimation models from YOLO would require extending the AIF interface.

**No inference timeout.** Inference nodes do not implement a watchdog or timeout on model inference. A model that hangs (e.g., due to GPU resource contention) will block the inference node indefinitely without publishing any error status.

**Training pipeline is not integrated with the NEPI build system.** The `nepi_ai_training` scripts are standalone Python programs intended to run on a development machine. They are included in a catkin package for distribution only; `catkin build` does not execute them.

## Decision Log

- 2026-03 — CLAUDE.md created — Initial developer reference, Claude Code authoring pass.
