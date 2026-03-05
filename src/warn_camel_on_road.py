"""
Camel-on-road warning: run camel detection + road segmentation per frame.
If a camel bbox overlaps the road mask above a threshold, print and overlay "CAMEL ON ROAD".
All outputs are saved locally under the project directory.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from paths import PROJECT_ROOT, RUNS_WARN, CAMEL_WEIGHTS_PATH, ROAD_WEIGHTS_PATH

# Default weights (local project paths; fallback to base models if missing)
DEFAULT_CAMEL_WEIGHTS = str(CAMEL_WEIGHTS_PATH)
DEFAULT_ROAD_WEIGHTS = str(ROAD_WEIGHTS_PATH)
FALLBACK_CAMEL = "yolov8n.pt"
FALLBACK_ROAD = "yolov8n-seg.pt"


def resolve_weights(path: str, fallback: str) -> str:
    """Return path if file exists, else fallback."""
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    return fallback


def get_road_mask_binary(results, frame_h: int, frame_w: int) -> np.ndarray:
    """
    Build a binary road mask (class 0) in frame size from segment results.
    If no masks or no class 0, return empty mask.
    """
    out = np.zeros((frame_h, frame_w), dtype=np.uint8)
    if not results or len(results) == 0:
        return out
    r = results[0]
    if r.masks is None or len(r.masks) == 0:
        return out
    masks = r.masks
    boxes = r.boxes
    for i in range(len(masks.data)):
        cls_id = int(boxes.cls[i].item()) if boxes is not None and boxes.cls is not None else 0
        if cls_id != 0:
            continue
        m = masks.data[i].cpu().numpy()
        m_resized = cv2.resize(m.astype(np.float32), (frame_w, frame_h))
        out = np.maximum(out, (m_resized > 0.5).astype(np.uint8))
    return out


def overlap_ratio(road_mask: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    """Road pixels inside bbox / bbox area. Clamps to frame bounds."""
    h, w = road_mask.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    crop = road_mask[y1:y2, x1:x2]
    road_pixels = int(np.sum(crop > 0))
    return road_pixels / area


def main():
    parser = argparse.ArgumentParser(description="Camel-on-road warning from video")
    parser.add_argument("--source", type=str, required=True, help="Video path")
    parser.add_argument("--camel-weights", type=str, default=DEFAULT_CAMEL_WEIGHTS)
    parser.add_argument("--road-weights", type=str, default=DEFAULT_ROAD_WEIGHTS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--overlap", type=float, default=0.15, help="Overlap threshold for warning")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--save", action="store_true", help="Save output to runs/warn/")
    parser.add_argument("--max-frames", type=int, default=None, help="Process only N frames (e.g. 30 for smoke test)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_file():
        print("Error: source file not found:", args.source, file=sys.stderr)
        sys.exit(1)

    cw = resolve_weights(args.camel_weights, FALLBACK_CAMEL)
    rw = resolve_weights(args.road_weights, FALLBACK_ROAD)

    print("Loading camel model:", cw)
    camel_model = YOLO(cw)
    print("Loading road model:", rw)
    road_model = YOLO(rw)
    print("Models loaded OK.")

    cap = cv2.VideoCapture(str(source.resolve()))
    if not cap.isOpened():
        print("Could not open video:", args.source, file=sys.stderr)
        sys.exit(1)

    out_video = None
    if args.save:
        save_dir = RUNS_WARN
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{source.stem}_warn.mp4"
        # Will set fourcc and size after first frame

    frame_idx = 0
    max_frames = args.max_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break

        h, w = frame.shape[:2]

        # Road segmentation -> binary mask (class 0)
        road_results = road_model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        road_mask = get_road_mask_binary(road_results, h, w)

        # Camel detection
        camel_results = camel_model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
        boxes = camel_results[0].boxes if camel_results and camel_results[0].boxes is not None else None

        any_camel_on_road = False
        if boxes is not None:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                ratio = overlap_ratio(road_mask, x1, y1, x2, y2)
                on_road = ratio >= args.overlap
                if on_road:
                    any_camel_on_road = True
                    print(f"Frame {frame_idx}: CAMEL ON ROAD overlap={ratio:.2f}")

                label = f"camel {ratio*100:.0f}%"
                if on_road:
                    label = f"camel ⚠ ON ROAD ({ratio*100:.0f}%)"
                color = (0, 0, 255) if on_road else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if on_road:
                    cv2.putText(frame, "CAMEL ON ROAD", (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Big red warning in center when any camel is on road
        if any_camel_on_road:
            text = "CAMEL ON ROAD"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.2
            thickness = 4
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cx = (w - tw) // 2
            cy = (h + th) // 2
            # Outline in black then red fill for visibility
            cv2.putText(frame, text, (cx - 1, cy), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (cx + 1, cy), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (cx, cy - 1), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (cx, cy + 1), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(frame, text, (cx, cy), font, font_scale, (0, 0, 255), thickness)

        if out_video is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_video = cv2.VideoWriter(str(out_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

        if out_video is not None:
            out_video.write(frame)

        if args.show:
            cv2.imshow("Camel on road", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()
        print("Saved:", out_path)
    if args.show:
        cv2.destroyAllWindows()
    print("Processed", frame_idx, "frames.")


if __name__ == "__main__":
    main()
