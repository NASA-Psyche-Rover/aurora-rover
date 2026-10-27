# Perception stack inventory (FROZEN: do not edit, rename, move or reformat)

Recorded by `tools/perception_freeze.py record`. Hashes are `git hash-object` (blob SHA-1).

## objdetec (MuJoCo Sim copy, used by the adapters)
`/Users/aayou/Desktop/Aayou/ASU/EPICS/NASA Psyche/Stage 3/MuJoCo Sim/psyche-objdetec`  branch `oak-d-lite-integration` @ `2a6a2e1094`

Pre-existing uncommitted changes (were already there, left as is):
```
M .DS_Store
```

| file | role | git blob hash | functions / classes |
|---|---|---|---|
| `main.py` | live detection + terrain-risk pipeline | `bdc2ad9c0c6d57ec0140ecfc88a5662abe32d1a6` | def draw_corner_inset<br>def show_sample_images<br>def main |
| `OAKD-obj-detec.py` | OAK-D Lite detection + real depth test | `877ac7348063e02e94650c2ac8c931741b7c02c3` | def draw_hud<br>def main |
| `src/camera_stream.py` | camera input (webcam / OAK-D Lite RGB + stereo mm) + metric object distance | `310a31a0f26b261327a7ce36e24d42475188ec6b` | def get_video_source<br>class OakDLiteCamera: __init__, read, close<br>def estimate_object_distances_mm<br>def nearest_object_distance_m |
| `src/decision.py` | navigation gate from risk + proximity + density | `2a98d94a54c5e284ec50893fc3e2d988592e6c2a` | def should_proceed |
| `src/detect.py` | object detection (YOLOv8) | `912dc66a7e086a548a0ddb5c30a8cdcc8e018350` | class Detector: __init__, detect_objects |
| `src/terrain_risk.py` | depth analysis (MiDaS-small ONNX) -> slope/roughness/drop risk | `e39ed723e76fdaceec7071ac9d2a4c75da73deb3` | class TerrainAnalyzer: __init__, _estimate_depth, get_risk_assessment |
| `src/utils.py` | detection post-processing (density, depth-sampled proximity), HUD drawing | `8d900db01b2b4b0e9e91f1df488a8ef62bba500a` | def draw_test_button<br>def get_sample_images<br>def compute_cluster_density<br>def estimate_object_proximity<br>def draw_alert |
| `scripts/terrain_demo.py` | terrain-risk live demo / benchmark | `b20f59af7065560ff5e3ee7a67acc4e70adbfba7` | def main |
| `scripts/quantize_profile.py` | MiDaS backend benchmark | `2b0b9d3d5889c52a23a91ce3552aa15809681ff5` | def pct<br>def file_mb<br>def bench<br>def print_row<br>def load_midas<br>def export_onnx<br>def export_openvino<br>def bench_pytorch<br>def bench_onnx<br>def bench_openvino<br>def main |
| `legacy/terrain_risk_cubli.py` | legacy Depth-Anything terrain risk (reference only) | `700e33250a958ab2dea71341615b65333e0a4f9e` | class TerrainAnalyzer: __init__, get_risk_assessment |
| `models/best.pt` | model weights | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | - |
| `models/yolov8n.pt` | model weights | `0db4ca4b4d25d68a4497220353592645ca4f71bf` | - |
| `models/midas_small.onnx` | model weights | `1a76f2316f5b43f59036c522f092a65fd9f1f400` | - |
| `models/midas_small.onnx.data` | model weights | `fde3def76b03e3062a9bd41af737bfcb18848e5c` | - |
| `models/midas_small_openvino/midas_small.xml` | model weights | `7532ac76fa95f66426c8ca0ac4ebfe939e08e29d` | - |
| `models/midas_small_openvino/midas_small.bin` | model weights | `9dea5aec4da901f0ae78ae497902bacaafb3ba90` | - |

## objdetec (~/psyche-objdetec, older fork)
`/Users/aayou/psyche-objdetec`  branch `oak-d-lite-integration` @ `ee3b909950`

Pre-existing uncommitted changes (were already there, left as is):
```
M main.py
```

| file | role | git blob hash | functions / classes |
|---|---|---|---|
| `main.py` | live detection + terrain-risk pipeline | `8ece172cfa88bc33a977504f14ece275a171ba2e` | def _safe_int<br>def draw_status_panel<br>def draw_button<br>def draw_boxes<br>def load_next_sample<br>def compute_density_safe<br>def should_proceed_simple<br>def main |
| `src/camera_stream.py` | camera input (webcam / OAK-D Lite RGB + stereo mm) + metric object distance | `d0598df35c3d812ae788eb9f8944bd72bca2d8fd` | def get_video_source |
| `src/decision.py` | navigation gate from risk + proximity + density | `db556087d80671cf7c597b02e090e15b76fa48f4` | def should_proceed |
| `src/detect.py` | object detection (YOLOv8) | `a566810f93b088b5eb3cc83e2704b2072d72a304` | class Detector: __init__, detect_objects |
| `src/terrain_risk.py` | depth analysis (MiDaS-small ONNX) -> slope/roughness/drop risk | `14fc1051dd32245786a58f7d76a6cb9b18c97d56` | class TerrainAnalyzer: __init__, get_risk_assessment |
| `src/utils.py` | detection post-processing (density, depth-sampled proximity), HUD drawing | `13d39f9fa3692043b5d30098a76da28dd6d115b5` | def draw_test_button<br>def get_sample_images<br>def compute_cluster_density<br>def draw_alert |
| `models/best.pt` | model weights | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | - |
| `models/yolov8n.pt` | model weights | `0db4ca4b4d25d68a4497220353592645ca4f71bf` | - |

## cubli_sensors (LiDAR / stereo depth / camera)
`/Users/aayou/cubli_sensors`  branch `main` @ `c293f971fa`

| file | role | git blob hash | functions / classes |
|---|---|---|---|
| `cubli_sensors.py` | simulated camera image processing, stereo depth analysis, LiDAR | `16db43838bc52eff2c68a2799a58d3e5e29c6fbc` | def _rz<br>def _turbo<br>def _put<br>class _Orbit: __init__, drag, zoom<br>class _Zoomer: __init__, drag<br>class CubliSensors: __init__, psyche, psyche, gravity, _orbit_targets, handle_key, on_mouse, head_pose, _cam_pose, _tic, _toc, _render_cam, rgb, depth, world, clear_lidar, _pattern, lidar, _push, _title, _rgb_view, _depth_view, _pixel_heights, _cube_in_sensor, _cloud_cam, _project, _cloud_view, _bev_view, _world_view, _tabbar, _footer, compose, close |
| `demo_sensors.py` | sensor demo; ray-based obstacle/drop check (Rover.check_target) | `b5121d75146ff59605359e447f0395d4a1964b77` | def cube_model<br>def build_xml<br>def rodrigues<br>class Rover: __init__, reset, center, _heights, ground_z, check_target, _apply, set_mast, refresh, busy, heading, press, update, settle, deploy_now<br>def banner<br>def hud<br>def status<br>def snapshot<br>def bench<br>def interactive<br>def main |
| `psyche_terrain.py` | demo terrain for the sensor module | `f82a4c045e48b929818580464b885350e14fc747` | def _fft_noise<br>def _sample<br>def _ss<br>class Terrain: __init__, _n, height, _grid, mesh_objs<br>def _icosphere<br>def _rock_shape<br>def _euler<br>def _merged_obj<br>def _png<br>def regolith_texture<br>def rock_texture<br>def star_texture<br>def build |
| `tools/import_cubli_stl.py` | CAD import for the sensor demo | `4cf41e070ada7a3680c6b555deddf8a850d8ad39` | def load_stl<br>def weld<br>def components<br>def classify<br>def decimate<br>def main |
| `assets/cubli_frame.obj` |  | `e77d4a6721a1035db908e7809c8a5e103f0aaa91` | - |
| `assets/cubli_wheel.obj` |  | `549f6b186649ebd8af23ee958b65075e81676b27` | - |
| `assets/cubli_motor.obj` |  | `fbace4a4c400fcc3b94ed988f5f009b74882ff98` | - |
| `assets/cubli_hardware.obj` |  | `09765bec1c29c9b7ebf0afac307a6f961a015965` | - |
