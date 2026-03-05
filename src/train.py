"""Train YOLOv8 detection (camels) and segmentation (roads) via Python API only.
All outputs (runs, weights) are saved locally under the project directory."""

from ultralytics import YOLO

from paths import PROJECT_ROOT

# --- Config (edit these) ---
TASK = "camel"  # "camel" | "road" | "both"
CAMEL_DATA = "camel_data.yaml"
ROAD_DATA = "road_data.yaml"
CAMEL_WEIGHTS = "yolov8n.pt"
ROAD_WEIGHTS = "yolov8n-seg.pt"  # place in project root or let Ultralytics download
IMGSZ = 960
EPOCHS = 100
BATCH = 8
# All training outputs saved locally under project
PROJECT = str(PROJECT_ROOT / "runs")
CAMEL_NAME = "camel_detect"
ROAD_NAME = "road_segment"
CAMEL_RESUME = None  # e.g. path to runs/detect/camel_detect/weights/best.pt
ROAD_RESUME = None   # e.g. path to runs/segment/road_segment/weights/best.pt


def train_camels():
    weights = CAMEL_RESUME if CAMEL_RESUME else CAMEL_WEIGHTS
    model = YOLO(weights)
    data_path = PROJECT_ROOT / CAMEL_DATA if (PROJECT_ROOT / CAMEL_DATA).exists() else CAMEL_DATA
    model.train(
        data=str(data_path),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT,
        name=CAMEL_NAME,
    )


def train_roads():
    weights = ROAD_RESUME if ROAD_RESUME else ROAD_WEIGHTS
    model = YOLO(weights)
    data_path = PROJECT_ROOT / ROAD_DATA if (PROJECT_ROOT / ROAD_DATA).exists() else ROAD_DATA
    model.train(
        data=str(data_path),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT,
        name=ROAD_NAME,
    )


if __name__ == "__main__":
    if TASK == "road":
        train_roads()
    elif TASK == "both":
        train_camels()
        train_roads()
    else:
        train_camels()
