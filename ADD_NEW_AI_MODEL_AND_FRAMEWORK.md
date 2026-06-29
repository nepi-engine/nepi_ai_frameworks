# Adding a New AI Model and Framework to NEPI

Runbook for turning a model file (e.g. an Ultralytics `.pt` or exported `.onnx`) into a
NEPI-discoverable AI model, creating a new framework adapter only when one doesn't already exist.
Written from the working `nepi_aif_yolo26` build (Ocean YOLO26 ONNX, 2026-06-28).

---

## 0. Decision tree — start here when given a file

1. **Identify the model family + file type.** Filename, the training framework, and extension
   (`.pt` = PyTorch checkpoint, `.onnx` = exported ONNX, `.engine` = TensorRT, `.hef` = Hailo).
2. **Does a framework adapter already exist** under `src/nepi_ai_frameworks/` and a model dir under
   `/mnt/nepi_storage/ai_models/<framework>/`? Current adapters: `yolov8`, `yolov11`, `yolo26`, `hailo`.
   - **Yes** → skip Part A. Just add a model (Part B), then deploy/verify (Part C).
   - **No**  → create a new adapter (Part A), then Part B and Part C.
3. A new *framework* is only needed when inference is genuinely different (different package, different
   output parsing, different file type that the existing node can't load). An Ultralytics `.onnx` that a
   YOLO framework can already load is just a **new model**, not a new framework.

---

## 1. How NEPI discovers and runs models (the contract you must satisfy)

`ai_models_mgr.py` (`src/nepi_engine/nepi_managers/scripts/`) drives everything; the helpers live in
`src/nepi_engine/nepi_sdk/src/nepi_sdk/nepi_aifs.py`:

1. **Framework discovery** — `getAIFsDict(params_path, api_path)` scans `*params*.yaml` in the installed
   share dir, keys each framework by its **`framework_name`**, and **purges** any whose `if_file_name`
   isn't present in the api dir.
2. **Adapter load + support check** — imports the IF class via `importAIFClass(...)`, instantiates it,
   and calls **`checkFrameworkSupport()`**. **If this returns `False` the framework is purged** — so its
   dependency checks must pass on the target device.
3. **Model discovery** — `getModelsDict()` → `loadModelsDict(framework_name, pkg_name, models_folder)`
   scans `*.yaml` in `/mnt/nepi_storage/ai_models/<framework>/` and **requires** these keys:
   `framework`, `weight_file`, `image_size`, `classes`; requires `framework.name == framework_name`;
   requires the weight file to exist on disk; reads `type` (`detection`).
4. **Launch** — `launchModelNode()` resolves `node_file_dict[model_type]` (`detection`), sets the
   `weight_file_path` / `param_file_path` ROS params, and launches
   `/opt/nepi/nepi_engine/lib/<pkg_name>/<node_file_name>`.

**Installed paths on the device (where the manager actually looks):**
- Framework params: `/opt/nepi/nepi_engine/share/nepi_aifs/`
- Adapter IF module: `/opt/nepi/nepi_engine/lib/python3/dist-packages/nepi_api/`
- Detection node: `/opt/nepi/nepi_engine/lib/<pkg_name>/`
- Models: `/mnt/nepi_storage/ai_models/<framework>/`

> Creating files in the workspace source tree is **not enough** — the package must be built/installed
> and the model files present on the **device** before the manager sees them.

---

## 2. Extract classes + image size from the model file (do this first)

Ultralytics embeds class names and input size in the model. The model config yaml (Part B) needs them
verbatim — **never fabricate class names.**

**Best method — on any machine with `ultralytics` + `onnxruntime` (e.g. the device):**

```bash
python3 - <<'PY'
import onnxruntime as ort, ast
path = "/path/to/model.onnx"
s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
meta = s.get_modelmeta().custom_metadata_map      # {'names': "{0:'Triton',...}", 'imgsz':'[640, 640]', ...}
names = ast.literal_eval(meta["names"])           # dict {idx: name}
shape = s.get_inputs()[0].shape                   # [1, 3, H, W]
print("imgsz:", meta.get("imgsz"), "input shape:", shape)
print("classes (yaml-ready, index order):")
for i in sorted(names): print("      - " + str(names[i]))
PY
```

For a `.pt`: `from ultralytics import YOLO; m = YOLO("model.pt"); print(m.names)`.

**Workstation fallback (no `onnxruntime`/`onnx` installed):** `pip install onnx` then
`onnx.load(path).metadata_props`, or parse the protobuf `metadata_props` strings directly. The keys to
find are `names` and `imgsz`. Also note `task` (`detect`) and `end2end` (NMS baked into the graph if True).

Record: **class list in index order**, **image width/height**, model variant, and whether it's `end2end`.

---

## 3. Part A — Create a new framework adapter (clone)

**Always clone `nepi_aif_yolov8`.** It is complete and correct.
**Do NOT clone `nepi_aif_yolov11`** — its detection node is missing the `self.model = YOLO(weight_path)`
load line and crashes in `processImage`. (Verify: its "Load Model" block ends at CUDA detection.)

```bash
FW=yolo26                                  # new framework name (lowercase)
Cap=Yolo26                                 # capitalized class prefix
SRC=src/nepi_ai_frameworks/nepi_aif_yolov8
DST=src/nepi_ai_frameworks/nepi_aif_$FW
cp -r "$SRC" "$DST"
find "$DST" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
mv "$DST/api/aif_yolov8_if.py"                     "$DST/api/aif_${FW}_if.py"
mv "$DST/scripts/nepi_ai_yolov8_detection_node.py" "$DST/scripts/nepi_ai_${FW}_detection_node.py"
mv "$DST/params/aif_yolov8_params.yaml"            "$DST/params/aif_${FW}_params.yaml"
# rename all identifiers (two patterns cover everything: lowercase + capitalized)
grep -rl --null -e 'yolov8' -e 'Yolov8' "$DST" | xargs -0 sed -i -e "s/Yolov8/$Cap/g" -e "s/yolov8/$FW/g"
grep -rn -e 'yolov8' -e 'Yolov8' "$DST" || echo "clean — no remnants"
```

This renames, consistently: package `nepi_aif_<fw>`, `framework_name: <fw>`, class `<Cap>AIF`,
files `aif_<fw>_if.py` / `nepi_ai_<fw>_detection_node.py`, detector `<Cap>Detector`,
`DEFAULT_NODE_NAME "ai_<fw>"`, `MODEL_FRAMEWORK "<fw>"`, `models_folder_name <fw>`,
`project(nepi_aif_<fw>)` in CMakeLists, `<name>` in package.xml, and the `catkin_install_python` target.

**File roles** (mirror of yolov8):
| File | Role |
|---|---|
| `api/aif_<fw>_if.py` | `<Cap>AIF` class: `checkFrameworkSupport`, `getModelsDict`, `launchModel`, `killModel` |
| `scripts/nepi_ai_<fw>_detection_node.py` | `<Cap>Detector` ROS node: loads model, `processImage`/`processFile`, publishes via `AiDetectorIF` |
| `params/aif_<fw>_params.yaml` | discovery config (`framework_name`, `node_file_dict.detection`, IF class names) |
| `CMakeLists.txt` / `package.xml` | catkin build + install targets |

**For ONNX models — add an `onnxruntime` check** in `checkFrameworkSupport` (alongside cv2/torch/ultralytics):
```python
        if supported == True:
            check='onnxruntime'
            if nepi_utils.check_module_available(check) == False:
                supported = False
                self.logger.log_warn("Framework failed check: " + check)
```

**No node code change is needed for ONNX.** `ultralytics.YOLO("model.onnx")` loads and runs an exported
ONNX exactly like a `.pt`, returning a `Results` object with `.boxes.cls/.xyxy/.conf`. `end2end=True`
(NMS-in-graph) does not change the returned structure.

**Recommended hardening** (the yolov8 template has a latent bug): in `processImage` and `processFile`,
initialize `ids = []; boxes = []; confs = []` *before* the inference `try`, so a failed inference returns
zero detections instead of raising `UnboundLocalError` at `for i, idf in enumerate(ids)`.

---

## 4. Part B — Create the model files

```bash
FW=yolo26
mkdir -p /mnt/nepi_storage/ai_models/$FW/
cp /path/to/source_weights.onnx /mnt/nepi_storage/ai_models/$FW/<descriptive_name>_640.onnx
```
Name the weight file descriptively and consistently (e.g. `ocean_yolo26_640.onnx`,
`common_objects_yolov8_640.pt`). Verify the copy: `md5sum source dest`.

**Model config yaml** — one per model, in the same folder. Template (keys are mandatory; classes verbatim
from §2 in index order):

```yaml
ai_model:

  framework:
    name: yolo26                 # MUST equal framework_name in the adapter's params yaml
  type:
    name: detection
  display_name:
    name: ocean_yolo26       # short label shown in the RUI
  description:
    name: Ocean YOLO26 marine object detector
  weight_file:
    name: ocean_yolo26_640.onnx   # exact filename in this folder
  image_size:
    image_width:
      value: 640
    image_height:
      value: 640
  classes:
    names:
      - Triton
      - boat
      # ... all classes, in index order 0..N-1
```

---

## 5. Part C — Validate, deploy, run

**Static validation (workstation, no ML deps needed):**
```bash
python3 -m py_compile src/nepi_ai_frameworks/nepi_aif_<fw>/scripts/nepi_ai_<fw>_detection_node.py
python3 -m py_compile src/nepi_ai_frameworks/nepi_aif_<fw>/api/aif_<fw>_if.py
# yaml passes the same checks loadModelsDict enforces:
python3 - <<'PY'
import yaml, os
f="/mnt/nepi_storage/ai_models/<fw>/<model>.yaml"; d=yaml.safe_load(open(f)); k=list(d)[0]
for r in ("framework","weight_file","image_size","classes"): assert r in d[k], r
assert os.path.exists(os.path.join(os.path.dirname(f), d[k]["weight_file"]["name"]))
print("yaml OK,", len(d[k]["classes"]["names"]), "classes")
PY
```

**Deploy to the device:** build/install the package so it lands in `/opt/nepi/...` (see §1 paths), and
copy the model files to `/mnt/nepi_storage/ai_models/<fw>/` on the device. Then the manager discovers it.

**Runtime deps on the device** (must be importable by the node's interpreter — `/usr/bin/python3`):
`.pt` models → `torch`, `ultralytics`, `cv2`; `.onnx` models → also `onnxruntime`.

**Smoke test on the device:**
```bash
YOLO_OFFLINE=1 python3 -c "
from ultralytics import YOLO; import numpy as np
m = YOLO('/mnt/nepi_storage/ai_models/<fw>/<model>.onnx', task='detect')
print('names:', m.names)
print('boxes:', len(m(np.zeros((640,640,3),'uint8'), verbose=False)[0].boxes))"
```
Expect the class dict and `boxes: 0` on a blank frame. Then restart the model node.

---

## 6. Runtime gotchas (learned the hard way)

- **`checkFrameworkSupport` (`find_spec`) ≠ actually runs.** It only confirms the package *folder* exists;
  it does **not** import it, so a broken native lib (below) passes discovery and only fails at inference.
  Always confirm with a real `import` / model load on the device.

- **onnxruntime vs `GLIBCXX` (the big one).** On Ubuntu 20.04 / JetPack 5 devices `libstdc++` maxes out at
  `GLIBCXX_3.4.28`. **onnxruntime ≥ 1.16** is built on `manylinux_2_28` and needs `GLIBCXX_3.4.29` →
  fails with `version 'GLIBCXX_3.4.29' not found ... onnxruntime_pybind11_state.so`. Fix: install a
  `manylinux2014` build, **≤ 1.15.1**, into the **node's** interpreter (not bare `pip3`, which may be a
  different env):
  ```bash
  sudo python3 -m pip uninstall -y onnxruntime onnxruntime-gpu
  sudo python3 -m pip install --no-deps "onnxruntime==1.15.1"
  python3 -c "import onnxruntime; print(onnxruntime.__version__, onnxruntime.get_available_providers())"
  ```
  Check the device ceiling: `strings /usr/lib/aarch64-linux-gnu/libstdc++.so.6 | grep GLIBCXX | sort -V | tail`.

- **Ultralytics auto-reinstalls onnxruntime at runtime (this is why the pin above "won't stick").** When
  CUDA is present, Ultralytics' ONNX backend runs `check_requirements(("onnx", "onnxruntime-gpu"))` and,
  with auto-install on, pip-installs a newer `onnxruntime-gpu` — re-triggering the `GLIBCXX_3.4.29` failure
  on every node launch (rapid retry loop in the logs). The detection node guards against this by setting,
  **before importing ultralytics**:
  ```python
  os.environ["YOLO_AUTOINSTALL"] = "false"   # don't let ultralytics pip-install at runtime
  os.environ["YOLO_OFFLINE"] = "true"
  ```
  If you ever load a model outside the node (manual test), use `YOLO_AUTOINSTALL=false` (or `YOLO_OFFLINE=1`)
  on the command line for the same reason.

- **CPU vs GPU on Jetson.** PyPI `onnxruntime==1.15.1` is **CPU-only** (slow). For GPU use the
  **NVIDIA Jetson Zoo `onnxruntime-gpu` wheel** matched to your JetPack + Python, or build a **TensorRT
  `.engine`** from the ONNX and point the model at that. `Can't initialize NVML` / `Failed to start ONNX
  Runtime with CUDA` mean the GPU stack isn't available to the process — resolve before expecting GPU.

- **`task='detect'`** silences the "Unable to automatically guess model task" warning for ONNX loads.

- **First load is slow.** Importing `ultralytics` pulls in `torch`; on embedded hardware the node's
  `__init__` import + model load + 10-frame self-test takes ~20–40s before it publishes. One-time, not a hang.

- **`YOLO_OFFLINE=1`** prevents Ultralytics first-run network calls (pip version check, settings sync,
  Arial font download) from stalling on devices with no/limited internet.

- **`pthread_setaffinity_np failed ... Invalid argument`** from onnxruntime is **harmless** thread-pinning
  noise on cpuset-restricted / Jetson environments — inference still completes.

---

## 7. Reference: the `yolo26` build (worked example)

- Adapter: `src/nepi_ai_frameworks/nepi_aif_yolo26/` (cloned from yolov8; onnxruntime check added; node hardened)
- Model: `/mnt/nepi_storage/ai_models/yolo26/ocean_yolo26_640.{onnx,yaml}`
- Source: Ocean `90_weights.onnx` — Ultralytics **YOLO26s**, `task=detect`, **640×640**, `end2end=True`
- Classes (12): `Triton, boat, buoy, channel marker, crane, dock, fish, marine-obstacles, oil rig, person, sailboat, ship`
- Device fix that made it run: `onnxruntime==1.15.1` (CPU) on `/usr/bin/python3` 3.8.
