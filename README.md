# Camel Near Road Detection (YOLOv8)

Train two YOLOv8 models: **camel detection** (boxes) and **road segmentation** (masks). Run both on video to get a warning when a camel appears on or near the road.

**All outputs are saved locally** in the project directory: trained weights under `runs/detect/` and `runs/segment/`, warning videos under `runs/warn/`. No cloud or remote storage is used.

## Data

- **camel_dataset/** — camel detection (images + labels). Single class: camel.
- **road_dataset/** — road segmentation, same layout. Single class: road.

Config: `camel_data.yaml`, `road_data.yaml`.

---

## Development Setup

**Requirements:** Python 3.11, GPU recommended (e.g. RTX 3060 + CUDA). CPU works but is very slow.

1. **Create and activate virtual environment** (project root):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   CMD:

   ```cmd
   .\.venv\Scripts\activate.bat
   ```

2. **Install dependencies:**

   ```powershell
   python -m pip install --upgrade pip
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install ultralytics opencv-python
   ```

   Or: `pip install -r requirements.txt` then install PyTorch CUDA separately as above.

3. **Train camel detector:**

   ```powershell
   python src/train.py
   ```

   Set `TASK = "camel"` at the top of `src/train.py`. Weights go to `runs/detect/camel_detect/weights/best.pt`.

4. **Train road segmentation:**

   Set `TASK = "road"` in `src/train.py`, then:

   ```powershell
   python src/train.py
   ```

   Weights: `runs/segment/road_segment/` (from `train.py` default). Override in the warn script with `--road-weights` if you use a different path.

5. **Run camel detection on a video:**

   Set `VIDEO_PATH` in `src/test_video.py`, then:

   ```powershell
   python src/test_video.py
   ```

---

## Warning: Camel On Road

Script **`src/warn_camel_on_road.py`** runs both models per frame and warns when a camel bbox **overlaps the road mask** above a threshold.

**Overlap logic:** For each detected camel bbox, compute  
`overlap_ratio = (road pixels inside bbox) / (bbox area)`.  
If `overlap_ratio >= --overlap` (default 0.15), the script prints and overlays **"⚠ CAMEL ON ROAD"** for that bbox.

**Usage example:**

```powershell
python src/warn_camel_on_road.py --source "C:\path\to\video.mp4" --show
```

With options:

```powershell
python src/warn_camel_on_road.py --source video.mp4 --overlap 0.2 --conf 0.25 --save
```

Outputs are saved locally to `runs/warn/`. Use `--max-frames 30` for a quick test.

---

## Layout

```
camel-yolo-near-road/
├── camel_dataset/
├── road_dataset/
├── runs/              # training & warn outputs (gitignored)
├── src/
│   ├── split_dataset.py
│   ├── train.py
│   ├── test_video.py
│   └── warn_camel_on_road.py
├── camel_data.yaml
├── road_data.yaml
└── README.md
```
