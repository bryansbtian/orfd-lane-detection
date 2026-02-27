# Off-Road Traversable Road Segmentation

## Quick Comparison

|                | YOLO26x (YOLOE-26x)                    | SAM3                                   | SAM2.1-L                               |
| -------------- | -------------------------------------- | -------------------------------------- | -------------------------------------- |
| **Approach**   | Real-time open-vocabulary segmentation | Foundation model, concept segmentation | Point-prompted foundation segmentation |
| **Speed**      | ~3–8 ms / image (H200)                 | ~30 ms / image (H200)                  | ~50–100 ms / image (H200)              |
| **Model size** | ~70 MB                                 | 3.4 GB                                 | ~224 MB                                |
| **Prompting**  | Text prompts (open-vocab)              | Text prompts (open-vocab)              | Point prompt (bottom-centre of frame)  |
| **Zero-shot**  | Good — open-vocab via text prompts     | Strong — 47.0 LVIS Mask AP             | Strong — SAM2.1 architecture           |
| **Video**      | Frame-by-frame                         | Temporal tracking built-in             | Frame-by-frame                         |
| **Weights**    | Auto-downloaded                        | Manual download required               | Auto-downloaded                        |

---

## 1 — Environment Setup (Conda)

### 1a. Create a conda environment

```bash
conda create -n offroad-seg python=3.11 -y
conda activate offroad-seg
```

### 1b. Install PyTorch

**CPU only:**

```bash
conda install pytorch torchvision cpuonly -c pytorch -y
```

**GPU (CUDA 12.1) — recommended for SAM3 and SAM2.1-L:**

```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 1c. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

> **Note on the CLIP package (SAM3 only):**
> If you already have a different `clip` package installed, replace it:
>
> ```bash
> pip uninstall clip -y
> pip install git+https://github.com/ultralytics/CLIP.git
> ```

---

## 2 — Download Model Weights

### YOLO26 (YOLOE-26)

Weights are **downloaded automatically** on first run. Nothing to do.

### SAM2.1-L

Weights are **downloaded automatically** by Ultralytics on first run (~224 MB). Nothing to do.

### SAM3

SAM3 weights require manual download:

1. Go to <https://huggingface.co/facebook/sam3> and click **"Request access"**.
2. Once approved, go to the `Files and versions` tab and download **`sam3.pt`** (~3.4 GB).
3. Place `sam3.pt` in the project root (same folder as `segment_road.py`).

---

## 3 — Running the Script

### Segment all test images and videos

```bash
python segment_road.py --model yolo26 --input test_data/
python segment_road.py --model sam21  --input test_data/
python segment_road.py --model sam3   --input test_data/
```

### Single image

```bash
python segment_road.py --model yolo26 --input test_data/orfd.png
python segment_road.py --model sam21  --input test_data/orfd.png
python segment_road.py --model sam3   --input test_data/orfd.png
```

### Single video

```bash
python segment_road.py --model yolo26 --input test_data/aa.mp4
python segment_road.py --model sam21  --input test_data/aa.mp4
python segment_road.py --model sam3   --input test_data/aa.mp4
```

### Save a JSON metrics report

```bash
python segment_road.py --model yolo26 --input test_data/ --report
python segment_road.py --model sam21  --input test_data/ --report
python segment_road.py --model sam3   --input test_data/ --report
```

Reports are saved as `output/<model>/metrics_<model>.json`.

### CLI Options

| Flag              | Default       | Description                                                           |
| ----------------- | ------------- | --------------------------------------------------------------------- |
| `--model`         | _(required)_  | `yolo26`, `sam21`, or `sam3`                                          |
| `--input`         | `test_data/`  | Image/video file or directory                                         |
| `--output`        | `output/`     | Base output directory — a `<model>` subfolder is created inside       |
| `--conf`          | `0.25`        | Detection confidence threshold                                        |
| `--prompts`       | built-in set  | Custom text prompts _(YOLO26 and SAM3 only — ignored for SAM2.1)_     |
| `--model-size`    | `x`           | YOLOE-26 size: `n` / `s` / `m` / `l` / `x` _(ignored for SAM models)_ |
| `--sam3-weights`  | `sam3.pt`     | Path to SAM3 weights _(ignored for YOLO26 and SAM2.1)_                |
| `--sam21-weights` | `sam2.1_l.pt` | Path to SAM2.1 weights — auto-downloaded if not present               |
| `--report`        | off           | Save JSON metrics report to the model output directory                |

### Custom text prompts (YOLO26 / SAM3 only)

```bash
python segment_road.py --model yolo26 --input test_data/ \
    --prompts "dirt trail" "mud road" "driveable terrain" "gravel path"
```

---

## 4 — Output

All outputs are written to a model-specific subfolder inside `--output`:

```
output/
  yolo26/
  sam21/
  sam3/
```

### Annotated images

`output/<model>/<filename>_<model>_road.<ext>`
— Green semi-transparent mask over the traversable road
— Cyan contour boundary
— HUD overlay with per-frame metrics

### Annotated videos

`output/<model>/<filename>_<model>_road.mp4`
— Same overlay per frame + per-frame HUD

---

## 5 — Metrics

Metrics are split into three groups. **Performance** and **Model output** are always reported. **Ground truth** metrics are computed automatically when label images are present (images only, videos are skipped). All groups are reported identically for all models.

### Performance (always reported)

| Metric              | How it is calculated                                  |
| ------------------- | ----------------------------------------------------- |
| `inference_time_ms` | `time.perf_counter()` around `model.predict()`, in ms |
| `fps`               | `1000 / inference_time_ms`                            |

### Model Output (always reported)

| Metric              | How it is calculated                                                            |
| ------------------- | ------------------------------------------------------------------------------- |
| `road_coverage_pct` | `road pixels / total pixels × 100` — what fraction of the image was marked road |
| `mean_confidence`   | Mean of the model's own confidence scores across all detected road instances    |
| `num_detections`    | Count of distinct road instances returned by the model                          |

### Video consistency (video only, no ground truth needed)

| Metric         | How it is calculated                                                                                  |
| -------------- | ----------------------------------------------------------------------------------------------------- |
| `temporal_iou` | `(mask_t ∩ mask_{t-1}) / (mask_t ∪ mask_{t-1})` — frame-to-frame mask overlap; 1.0 = perfectly stable |

### Ground truth (images only, requires `test_data/labeled/`)

Computed by comparing the predicted mask against the hand-labelled image. Label pixels: **white (255) = road**, **black (0) = non-road**, **gray (128) = void** (excluded from all calculations).

Let `TP`, `TN`, `FP`, `FN` be counts of true/false positives/negatives over all non-void pixels:

| Metric           | Formula                            | What it tells you                                 |
| ---------------- | ---------------------------------- | ------------------------------------------------- |
| `iou`            | `TP / (TP + FP + FN)`              | Overlap between predicted and actual road area    |
| `precision`      | `TP / (TP + FP)`                   | Of pixels called road, how many actually are road |
| `recall`         | `TP / (TP + FN)`                   | Of actual road pixels, how many were found        |
| `f1`             | `2 × precision × recall / (P + R)` | Harmonic mean of precision and recall             |
| `pixel_accuracy` | `(TP + TN) / (TP + TN + FP + FN)`  | Overall per-pixel classification accuracy         |

The console prints `mean / min / max` for each metric across all images, and a per-image line with `IoU` and `F1` for quick scanning.
