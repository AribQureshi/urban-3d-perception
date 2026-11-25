# Monocular Metric 3D Perception and Collision-Risk Assessment for Urban Driving

**Technical report** · Arib Qureshi, M.Tech Artificial Intelligence, MPSTME, NMIMS

## Abstract

LiDAR gives autonomous vehicles accurate 3D perception but is expensive. This project asks how much 3D understanding of an urban road scene can be recovered from a single, uncalibrated dashcam. The system detects and tracks road users (YOLOv8 + ByteTrack) and estimates per-pixel metric depth with a Vision Transformer (Depth Anything V2). It then lifts every object into camera-frame 3D coordinates with a pinhole model and smooths each trajectory with a constant-velocity Kalman filter whose measurement noise grows with distance. Time-to-collision (TTC) gives a per-object risk level, and an Isolation Forest flags tracks with unusual behaviour without any labels. A RANSAC ground-plane fit on the reconstructed point cloud estimates the camera's height above the road, which acts as a ground-truth-free check on the depth scale. A configurable ego-vehicle mask removes the car's own dashboard and the driver from detection and reconstruction. On an 11-second 4K urban clip filmed from inside the car, the Kalman filter reduces frame-to-frame distance jitter by **45%** and speed noise by **90%** compared with raw monocular depth, and the pipeline runs at **4.5 FPS** on an Apple M5 laptop. The ground-plane check estimates a camera height of **2.36 m**, well above the expected 1.2 to 1.4 m, which indicates that the depth model overestimates distance on this clip.

## 1. Problem

Given a monocular video with no calibration, for every road user in every frame estimate:
1. its identity over time,
2. its 3D position in metres relative to the ego vehicle,
3. its velocity and time to collision, and
4. whether its behaviour is unusual compared with the rest of the scene.

## 2. Method

### 2.1 Detection and tracking
YOLOv8n (COCO-pretrained) detects six road-user classes: person, bicycle, car, motorcycle, bus and truck. ByteTrack associates detections across frames with a Kalman-predicted IoU match. It also uses low-confidence boxes in a second matching stage, which keeps IDs through partial occlusion.

### 2.2 Metric monocular depth
Depth Anything V2 Metric-Outdoor (ViT-S encoder, DPT decoder, fine-tuned on Virtual KITTI 2) predicts depth in metres for every pixel. The prediction is resized bilinearly to the input resolution.

### 2.3 Lifting to 3D
Without calibration, the focal length comes from an assumed horizontal field of view:

  f = (W / 2) / tan(HFOV / 2),   c = (W/2, H/2)

An object's distance Z is the **median depth over the central 50% of its bounding box**. This rejects background pixels that fall inside the box near its edges (unit-tested). The 3D centre is then

  X = (u − cₓ)·Z / f,   Y = (v − c_y)·Z / f,

and metric width and height follow from box size × Z / f.

### 2.4 Kalman filtering with depth-dependent noise
Each track has a constant-velocity Kalman filter over the ground-plane state **s** = [X, Z, Vₓ, V_z]. Process noise follows the discrete white-noise acceleration model (σₐ = 2 m/s²). Monocular depth error grows roughly in proportion to distance, so the measurement noise is

  σ_Z = max(0.3 m, 0.05·Z),   σ_X = max(0.15 m, 0.025·Z).

A 1 m surprise at 50 m therefore moves the estimate less than the same surprise at 5 m (unit-tested). Velocities are used only after three updates.

### 2.5 Risk and anomaly detection
TTC = Z / closing speed, defined when the object approaches faster than 0.3 m/s. Objects inside the ego lane corridor (|X| < 1.8 m) are graded CRITICAL (TTC < 1.5 s or Z < 5 m) or HIGH (TTC < 3 s or Z < 10 m). Any object with TTC < 5 s or Z < 15 m is at least MEDIUM. Each track is summarised by five standardised features: min and mean distance, maximum closing speed, mean lateral speed and minimum TTC. An Isolation Forest (contamination 0.1) then isolates the outlying tracks.

### 2.6 Scene reconstruction and depth-scale check
The depth map of the most critical frame is back-projected into a coloured point cloud. A RANSAC plane (300 iterations, 0.15 m threshold, seeded from image rows below 55% of the height and above the ego mask, refined by SVD) segments the road. Only candidate planes whose normal is within 30° of the camera's vertical axis are accepted. Without this constraint a building façade or the back of a vehicle can hold more inliers than the road, and in our first run RANSAC returned a plane facing the camera. The plane offset |d| is the camera height above the road. Dashcams are mounted 1.2 to 1.8 m high, so this number checks whether the metric depth scale and the assumed field of view are consistent, without any ground truth.

### 2.7 Ego-vehicle mask
A camera inside the car sees its own dashboard, bonnet or roof, and the driver. YOLOv8 detects the driver's arms as a *person* about 4 m away in the ego lane, which raises false CRITICAL alerts. The configuration therefore defines image bands at the top and bottom of the frame that belong to the ego vehicle (`ego_mask.top_frac`, `ego_mask.bottom_frac`). A detection is discarded when more than 20% of its box height lies inside a band, and masked pixels are excluded from the point cloud and the ground fit. A person standing just in front of the car is still kept, because the dashboard hides their lower body and their visible box ends at the mask edge.

## 3. Experiments

**Data.** An 11.3 s urban drive (3840×2160, 29.97 fps, 338 frames) filmed from the driver's position inside the car. [Source and city: fill in.] The dashboard, steering wheel and the driver's hands cover the bottom 26 to 31% of every frame, and the camera is not rigidly mounted: the windshield edge moves between 69% and 74% of the frame height. Every 2nd frame was processed (169 frames) with `configs/cockpit.yaml`, which masks the bottom 32% of the image.

**Hardware.** Apple M5 laptop with 16 GB of memory. Depth inference runs in FP32 on the Apple GPU (PyTorch MPS backend); YOLOv8n runs on the CPU (Ultralytics default). The Colab notebook runs the same code on a T4 GPU with FP16 depth.

**Metrics.** Ground-truth 3D boxes are not available for an arbitrary dashcam clip, so we use measurable proxies:
- *Distance jitter*: mean |Δ²Z| per frame, averaged over tracks with at least 6 observations. Real vehicles cannot change their acceleration abruptly, so lower jitter means more physically plausible estimates.
- *Speed noise*: standard deviation of the estimated closing speed.
- *Camera height*: from the RANSAC plane, compared with the physical mounting range.
- *Latency*: mean wall-clock time per pipeline stage.

## 4. Results

**Latency per stage**

| Stage | Mean latency (ms) | Share |
|---|---:|---:|
| Detection + tracking (YOLOv8n + ByteTrack, CPU) | 19.6 | 13% |
| Metric depth (Depth Anything V2 ViT-S, MPS) | 133.6 | 87% |
| 3D lifting + Kalman + risk | 0.1 | <1% |

End to end, including video decoding and writing, the pipeline processes 4.5 frames per second. Depth estimation is the bottleneck.

**Ablation: Kalman filter vs raw monocular depth**

| Distance estimate | Jitter (m) | Speed noise (m/s) |
|---|---:|---:|
| Raw per-frame depth | 3.782 | 45.21 |
| Kalman filtered (ours) | 2.072 | 4.69 |
| **Reduction** | **45.2%** | **89.6%** |

Evaluated on the 8 tracks with at least 6 observations. Jitter falls on 7 of the 8 tracks. The exception is a car (track 72) seen for only six frames, too few for the filter to converge. Raw speed noise of 45 m/s shows that differentiating per-frame monocular depth gives unusable velocities; filtering is required before TTC is meaningful.

**Scene statistics.** 28 tracked road users (13 cars, 14 persons and 3 bicycles; two tracks changed class during tracking). The ego mask discarded 129 detections of the driver and the dashboard; before it was added, the driver's arms alone produced 12 CRITICAL observations. Of the 121 remaining risk observations, 107 are LOW, 7 MEDIUM, 5 HIGH and 2 CRITICAL. All HIGH and CRITICAL observations come from one car (track 491) about 40 to 60 m away, whose estimated closing speed (up to 28 m/s) is inflated by depth noise. They are false alarms. The Isolation Forest flagged one track (482), the most distant car (77 m) with no relative motion. With only 11 tracks long enough to score, this flag reflects distance rather than unusual behaviour.

**Depth-scale check.** The RANSAC road plane is tilted 1.9° from the camera's horizontal axis, and 9% of reconstructed points lie on it. Its offset gives an estimated camera height of **2.36 m**. A camera at the driver's eye level should be about 1.2 to 1.4 m above the road, so the depth scale is too large by a factor of roughly 1.7 to 2.0. The assumed field of view cannot explain this: the height estimate is (v − c_y)·Z / f, so a wider true field of view (smaller f) would make it even larger. The likely cause is a domain gap between this scene, seen through a windshield, and the Virtual KITTI training data of the metric depth model. All absolute distances and TTC values in this run should therefore be read as overestimates. A single correction factor of about 0.55 would restore the expected camera height.

## 5. Discussion and limitations
- **Calibration.** The assumed field of view scales all lateral positions. A calibrated camera (e.g. KITTI) removes this error, and the camera-height check shows when the assumption is wrong.
- **Depth at range.** Monocular depth degrades beyond about 40 m. The depth-dependent Kalman noise reduces this effect but cannot remove it.
- **Ego-motion.** Velocities are relative to the moving camera. That is the right quantity for TTC, but not for absolute object speed.
- **Velocity on young tracks.** Closing speed is trusted after three updates, which is too early at the far end of the depth range. This caused the only false alarms in our clip (track 491). A longer warm-up, or a minimum on the velocity covariance before risk is assigned, would remove them.
- **Ego mask.** The mask is a fixed band, and the camera in our clip moves. A per-frame segmentation of the ego vehicle would be more robust.
- **Sample size.** One 11-second clip with 8 evaluable tracks; the numbers show the behaviour of the method, not a benchmark.
- **Evaluation.** Proxy metrics show stability, not accuracy. The next step is quantitative evaluation on KITTI tracking sequences against LiDAR ground truth (absolute relative depth error, 3D centre error, MOTA).

## 6. Future work
1. Evaluate on KITTI with LiDAR ground truth.
2. Geo-reference objects with GPS/IMU and aggregate risk events into an H3 hexagon hotspot map.
3. Replace the 2D box + median depth with a 3D box regressor, and test RT-DETR (a transformer detector) against YOLOv8.

## References
1. Jocher et al., *Ultralytics YOLOv8*, 2023.
2. Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022.
3. Yang et al., *Depth Anything V2*, NeurIPS 2024.
4. Fischler & Bolles, *Random Sample Consensus*, Communications of the ACM, 1981.
5. Liu, Ting & Zhou, *Isolation Forest*, ICDM 2008.
6. Kalman, *A New Approach to Linear Filtering and Prediction Problems*, 1960.
