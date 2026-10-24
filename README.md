<div align="center">

# 🚀 AURORA

**Autonomous Unit for Reaction-wheel Orientation & Real-time Analytics**

An autonomous rover concept for the surface of asteroid **16 Psyche**, built by ASU's NASA Psyche Autonomous Rover Team 1.

![NASA Psyche Capstone](https://img.shields.io/badge/NASA%20Psyche-Capstone-orange)
![ASU EPICS](https://img.shields.io/badge/ASU-EPICS-8C1D40)
![NASA Team Excellence Award](https://img.shields.io/badge/NASA%20Team%20Excellence%20Award-2025%20%7C%202026-gold)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![ESP32](https://img.shields.io/badge/ESP32-E7352C?logo=espressif&logoColor=white)
![NVIDIA Jetson](https://img.shields.io/badge/Jetson%20Orin%20Nano-76B900?logo=nvidia&logoColor=white)
![ROS 2](https://img.shields.io/badge/ROS%202-Humble-22314E?logo=ros&logoColor=white)

[Project page on the NASA Psyche capstone site](https://psyche.ssl.berkeley.edu/get-involved/capstone-projects/capstone-projects-platinum-class/robotic-explorer-for-hypothesized-surfaces-asu-b/)

</div>

---

## The problem

NASA's Psyche spacecraft reaches the metal-rich asteroid 16 Psyche in 2029, but it only **orbits**. Nothing is planned for the surface. Driving there is hard:

- **Gravity of ~0.144 m/s²**: wheels barely get traction, and a careless push can launch a rover off the surface.
- **Near-zero light**: cameras alone aren't enough.
- **Metal-rich, unknown terrain**, with communication delays that rule out joystick driving.

AURORA prototypes a rover that can **see** the terrain and **move** across it on its own under those constraints.

## How it works

| | |
|---|---|
| **Mobility: the Cubli** | A ~17 cm cube with three internal reaction wheels. Spinning a wheel up and braking it hard tips the cube over an edge onto its next face, so it "rolls like a die" without wheels touching the ground. |
| **Perception: P.A.R.** | A sensor testbed with an OAK-D Lite stereo camera (YOLO object detection + metric depth), a Livox Mid-360 LiDAR, and ultrasonic sensors, all running on an NVIDIA Jetson Orin Nano. |
| **Decision** | Measures real distances to rocks, drops, and steep slopes in the rover's path and outputs **PROCEED / CAUTION / STOP**. |

## Repository layout

```text
perception/   Object detection, stereo depth, terrain risk, mapping & navigation decisions (Python)
dashboard/    Live web dashboard: LiDAR overlay, object detection and 3D point cloud on one screen
electrical/   Cubli KiCad schematics + ESP32 bring-up firmware
```

Each folder has its own README with setup and run instructions.

## Project timeline

| Stage | Semester | What happened |
|---|---|---|
| **1: SCOUT** | Fall 2025 | Tethered mobility concept, initial autonomy and sensor research, LiDAR + ultrasonic obstacle avoidance |
| **2: Cubli + P.A.R.** | Spring 2026 | Chose a MINERVA-inspired reaction-wheel cube; built the P.A.R. sensor prototype; finalized Cubli schematics in KiCad |
| **3: Integration** | Fall 2026 (now) | Building a Cubli test rover and a sensor testbed; fusing LiDAR point clouds with the camera; LiDAR SLAM (FAST-LIO on ROS 2) |

## Results so far

- Static object detection validated at **88–92% confidence**
- LiDAR drop detection stops the rover **1.5–2.2 ft** before an edge
- Moving perception from a Raspberry Pi (8–10 FPS) to the Jetson removed the frame-rate bottleneck
- **NASA Team Excellence Award**, 2025 and 2026

## Team (Fall 2026)

| | |
|---|---|
| **Aayoush Iyer** | Team Lead · Software Lead |
| **Vivian Lee** | Mechanical Sub-Team Lead |
| **Rahul Kalikota** | Electrical Sub-Team Lead |
| Saanvi Kakde · Diya Shrivastava | Software |
| Julia Yap · Aarin Niranjane | Electrical |
| Alexandra Jacapraro · Joshua Raya · Atharva Manjarekar · Vrishank Singh | Mechanical |

Built through Arizona State University's EPICS program as part of the NASA Psyche capstone program.

> This repository contains only the team's own work. No confidential Psyche Mission information is included.
