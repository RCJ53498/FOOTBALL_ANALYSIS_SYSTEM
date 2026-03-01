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

## 🎬 Final Output

Check out the output showing the player tracking, speed estimation, team assignment, and dynamic analysis in action:

<video src="https://github.com/RCJ53498/FOOTBALL_ANALYSIS_SYSTEM/raw/main/output_videos/Final_Output.mp4" controls="controls" style="max-width: 100%;">
  Your browser does not support the video tag.
</video>
