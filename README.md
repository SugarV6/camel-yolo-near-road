# Camel Near Road Detection (YOLOv8)

## Overview

This project trains a YOLOv8 model to detect camels on roads and aims to trigger a warning when a camel is near the road area.

## Dataset

- Source: Kaggle camel object detection dataset
- Total images: 166 image-label pairs
- Train/Validation split: 80/20
- Format: YOLO format (.png + .txt)

## Model

- Model: YOLOv8n (pretrained)
- Epochs: 80
- Image size: 640
- Device: CPU
- mAP50: ~0.88

## Project Structure

```
camel-yolo-near-road/
│
├── dataset/
│   ├── images/
│   ├── labels/
│
├── raw/
├── src/
│   ├── split_dataset.py
│   └── train.py
│
├── data.yaml
└── README.md
```

## How to Run

1. Install dependencies:

   ```
   pip install ultralytics opencv-python
   ```

2. Train:

   ```
   python src/train.py
   ```

3. (Planned) Detection with warning script

## Future Work

- Camel near road warning logic
- Add more data
- Improve night detection
- GPU training
