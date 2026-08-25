# monocular-vla

Low-cost Vision-Language-Action (VLA) system for mobile robotics, using a single laptop webcam as the sole visual sensor.

**Status:** Phase 1 - Basic vision (color detection, object tracking, pixel↔tabletop mapping)

## Overview

This project investigates whether a functional VLA pipeline — perception, language grounding, task planning, and physical action — can be built end-to-end on a monocular (single-camera), low-cost setup (~Rp150,000–300,000 / ~$10–20 hardware budget), without relying on depth sensors, robot arms, or GPU workstations.

The camera is fixed/overhead, observing a workspace rather than mounted on the robot, to isolate the VLA reasoning problem from harder concurrent problems like ego-motion and self-localization.

## Architecture

Three strictly separated layers:

```
Layer 1 — Perception    : Webcam → OpenCV → object pixel coordinates
Layer 2 — Intelligence  : Coordinates + language instruction → VLM/planner → semantic action sequence
Layer 3 — Control       : Semantic action → low-level motor command → ESP32
```

The language/planning layer only ever outputs semantic actions (`GOTO(target)`, `PICK(object)`) — never raw motor commands. This keeps the simulator and physical robot backends interchangeable behind the same Action API.

## Roadmap

| Phase | Goal |
|---|---|
| 0 | Environment setup |
| 1 | Basic vision (color detection, object tracking, pixel↔tabletop mapping) |
| 2 | Vision-language grounding |
| 3 | Simulated robot (PyGame) + Action API |
| 4 | Full VLA pipeline in simulation |
| 4.5 | Multi-step instructions |
| 5 | Real-world perception, pointer-only actions |
| 5A/5B | Physical robot (ESP32) integration |
| 6 | (Optional) physical manipulation / gripper |
| 7 | ROS2 migration (wraps the working system, doesn't replace it) |
| 8 | Research benchmark: robustness to unseen objects, lighting, viewpoint, instruction paraphrasing |

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Verify webcam access:

```
python scripts/fase0_webcam_check.py
```

## Requirements

- Python 3.11+
- Webcam
- (Later phases) ESP32 dev board, DC motors + driver, VLM API access

## License

TBD
