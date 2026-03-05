# camel-yolo-near-road: YOLOv8 camel detection + road segmentation (https://github.com/SugarV6/camel-yolo-near-road)
"""Run camel detection + road segmentation on video; overlay and optionally filter camels on road."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# --- Config ---
VIDEO_PATH = r"C:\Users\sa888\Downloads\OurTour encounter a caravan of camels as they drive their motorhome around Morocco.mp4"
CAMEL_WEIGHTS = "runs/camel_detect/weights/best.pt"
ROAD_WEIGHTS = "runs/road_segment/weights/best.pt"
CONF = 0.25
FILTER_ON_ROAD = True  # only count camel if box center is inside road mask
SHOW_ON_ROAD = True   # print "ON ROAD" when camel center is inside road mask


def filter_on_road(box_xyxy, mask):
    """Return True if box center (x_center, y_center) is inside road mask."""
    if mask is None or mask.size == 0:
        return True
    x1, y1, x2, y2 = map(int, box_xyxy)
    xc = (x1 + x2) // 2
    yc = (y1 + y2) // 2
    h, w = mask.shape[:2]
    if 0 <= yc < h and 0 <= xc < w:
        return bool(mask[yc, xc])
    return False


def main():
    camel_model = YOLO(CAMEL_WEIGHTS)
    road_model = YOLO(ROAD_WEIGHTS)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Could not open video:", VIDEO_PATH)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Road segmentation
        road_results = road_model.predict(frame, conf=CONF, verbose=False)
        road_mask = None
        if road_results and len(road_results[0].masks) > 0:
            road_mask = road_results[0].masks.data[0].cpu().numpy()
            h, w = frame.shape[:2]
            m = cv2.resize(road_mask.astype(np.uint8), (w, h))
            overlay = frame.copy()
            overlay[m > 0] = overlay[m > 0] * 0.5 + (0, 128, 0)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Camel detection
        camel_results = camel_model.predict(frame, conf=CONF, verbose=False)
        camels_on_road = 0
        if camel_results and camel_results[0].boxes is not None:
            boxes = camel_results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                if FILTER_ON_ROAD and road_mask is not None:
                    if not filter_on_road(xyxy, road_mask):
                        continue
                camels_on_road += 1
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "camel", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if camels_on_road > 0:
            print("CAMEL DETECTED")
            if SHOW_ON_ROAD and FILTER_ON_ROAD and road_mask is not None:
                print("ON ROAD")

        cv2.imshow("camel + road", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
