# ⚽ FOOTBALL ANALYSIS SYSTEM

## Introduction

This project implements a comprehensive computer vision system designed for automated football (soccer) analysis. It leverages advanced tracking and detection models (YOLOv8) to identify players, the ball, and teams, and then uses specialized modules to estimate speed, distance, and analyze passing networks.

The system is organized into several key components to deliver a full-stack analysis pipeline for video input.

## Features

* **Player & Ball Tracking:** Robust multi-object tracking to maintain identity across frames.
* **Team Assignment:** Automatically assigns detected players to Team 1 or Team 2.
* **Speed & Distance Estimation:** Calculates player speed and distance covered using estimated camera motion and homography.
* **Passing Network Analyzer:** Generates graphical representations of passing connections and effectiveness between teammates.
* **Video Output:** Generates annotated videos showing detections, tracks, and analysis overlays.

## ⚙️ Prerequisites

Before running the project, ensure you have the following installed:

* Python (3.8+)
* Git (for cloning)
* Anaconda/Miniconda (Recommended for environment management)

### Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone [YOUR REPO URL HERE]
    cd FOOTBALL_ANALYSIS_SYSTEM
    ```
2.  **Create and activate the environment:**
    ```bash
    conda create -n football-env python=3.9
    conda activate football-env
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # If requirements.txt is missing, list the main dependencies here:
    # pip install opencv-python numpy pandas ultralytics
    ```

## 📂 Project Structure

The key components of the system are housed in the following directories:

| Directory | Description |
| :--- | :--- |
| `input_videos/` | Placeholder for raw video files (ignored by Git, but needed for input). |
| `models/` | **[CRITICAL]:** Stores pre-trained weight files (e.g., `yolov8x.pt`, `best.pt`). **These are ignored by Git and must be [DOWNLOADED SEPARATELY / PLACED HERE].** |
| `output_videos/` | Generated analysis videos (ignored by Git). |
| `output_images/` | Generated network graphs and output images. |
| `passing_network_analyzer/`| Core logic for analyzing player positions and generating passing networks. |
| `speed_and_distance_estimator/` | Module for calculating kinematic data using camera movement compensation. |
| `team_assigner/` | Logic for grouping players into their respective teams. |
| `utils/` | Helper functions (e.g., `bbox_utils.py`, `video_utils.py`). |

## 🚀 Usage

### 1. Place Input File and Models

1.  Place your raw video file (e.g., `08fd33_4.mp4`) into the `/input_videos/` directory.
2.  **Download all necessary model weights** (e.g., `yolov8x.pt`, `best.pt`) and place them in the `/models/` directory.

### 2. Run the Analysis Pipeline

Execute the main script from the root directory:

```bash
python main.py
# OR: python yolov8_inference.py --input_video input_videos/08fd33_4.mp4
