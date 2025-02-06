# Vehicle Speed Estimation

## Menu
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Arguments](#arguments)
- [Project Approach](#project-approach)
- [Performance Comparison](#performance-comparison)
- [Future Improvements](#future-improvements)
- [Cloud Deployment Strategy (AWS)](#cloud-deployment-strategy-aws)
- [Output](#output)
  - [Annotated Videos](#annotated-videos)
  - [CSV Reports](#csv-reports)
- [Docker Setup](#docker-setup)
- 
## Overview
This project performs vehicle detection, tracking, and speed estimation using the latest YOLO (`yolo11n.pt`) model, in combination with the Supervision library for bounding box manipulation and annotation, and OpenCV for video processing.

## Features
- **Object Detection**: Detects vehicles such as cars, motorcycles, buses, and trucks in video frames.
- **Object Tracking**: Uses ByteTrack to assign unique IDs and track vehicles across frames.
- **Perspective Transformation**: Uses a homography transformation to map image-space coordinates to real-world measurements.
- **Speed Estimation**: Calculates vehicle speeds in km/h by measuring movement across transformed real-world coordinates.
- **Annotated Video Output**: Saves annotated videos with detection, tracking, and speed overlays.
- **CSV Logging**: Exports vehicle speed statistics to CSV.

## Installation
### Prerequisites
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Supervision

### Setup
```bash
# Clone the repository
git clone https://github.com/SiuPuiCheung/project.git

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Running the Script
The main script provides multiple functionalities: detection, tracking, and speed estimation.
```bash
python main.py --video_path path/to/video.mp4 --task speed --show False
```

### Arguments
| Argument       | Description |
|---------------|-------------|
| `--video_path` | Path to the input video (default: videos/video_01.mp4). |
| `--task`       | Task to perform (`detection`, `tracking`, `speed`)  (default: speed). |
| `--conf`       | Confidence threshold for object detection (default: 0.4). |
| `--iou`        | Intersection Over Union (IoU) threshold (default: 0.7). |
| `--show`       | Display video while processing (`True` or `False`) (default: False). |

## Project Approach
1. **Video Frame Acquisition**: Reads frames from the input video.
2. **Object Detection**: YOLO is applied to detect vehicles (cars, motorcycles, buses, and trucks).
3. **Object Tracking**: ByteTrack assigns unique IDs and tracks vehicles across frames.
4. **Perspective Transformation**: Uses a manually calibrated homography transformation to map image coordinates to real-world coordinates.
5. **Speed Calculation**:
   - Calculates the distance traveled over time (in feet per second), converts to km/h.
   - Saves annotated videos and speed statistics to CSV files.

## Performance Comparison
| Approach | Pros | Cons |
|----------|------|------|
| **Using Only Ultralytics** | Simple, built-in tracking and speed estimation, supports GPU | Limited control over bounding boxes, no homography transformation |
| **YOLO + Supervision** | Flexible bounding box manipulation, supports homography transformation | More complex setup, manual anchor point calibration |

## Future Improvements
1. **Multithreading/Multiprocessing**: Improve efficiency by utilizing multiple CPU/GPU cores.
2. **Automated Anchor Point Detection**: Use Hough Line Transform for automatic point detection.
3. **Testing Functions**: Develop debugging tools.
4. **Hybrid Model**: Combine YOLO with other tracking techniques for improved accuracy.
5. **Stereo Vision**: Use multiple camera angles for 3D vehicle positioning.

## Cloud Deployment Strategy (AWS)
1. Store videos in S3, trigger processing via Lambda.
2. Use EC2 GPU instance to run the model.
3. Save annotated videos and CSV logs to S3.
4. Send SNS alerts when processing is complete.
5. Monitor pipeline using CloudWatch.

## Output
### Annotated Videos
- Detection output: `output/detection_YYYYMMDD_HHMMSS.mp4`
- Tracking output: `output/tracking_YYYYMMDD_HHMMSS.mp4`
- Speed estimation output: `output/speed_YYYYMMDD_HHMMSS.mp4`

### CSV Reports
- Speed statistics: `output/speed_YYYYMMDD_HHMMSS.csv`

## Docker Setup
```bash
# Clone the repository
git clone https://github.com/SiuPuiCheung/project.git

# Build the Docker image
docker build -t <image_name> .

# Run the Docker container
docker run <image_name>
