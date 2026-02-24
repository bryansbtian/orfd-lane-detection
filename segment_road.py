#!/usr/bin/env python3
"""
Off-Road Traversable Road Segmentation
=======================================
Segment the traversable road in off-road environments using either:
  - SAM3   — Meta's Segment Anything Model 3 (open-vocabulary concept segmentation)
  - YOLO26 — Ultralytics YOLOE-26 (open-vocabulary instance segmentation)

Both backends accept the same text prompts and report identical metrics so you
can make a direct, apples-to-apples comparison.

Usage
-----
  python segment_road.py --model yolo26 --input test_data/
  python segment_road.py --model sam3   --input test_data/

  python segment_road.py --model yolo26 --input test_data/orfd.png
  python segment_road.py --model yolo26 --input test_data/aa.mp4 --model-size s

  # Custom prompts
  python segment_road.py --model yolo26 --prompts "dirt trail" "mud road" "gravel path"

  # Save JSON metrics report
  python segment_road.py --model yolo26 --report

Notes
-----
- YOLO26 weights are downloaded automatically on first run.
- SAM3 weights (sam3.pt, ~3.4 GB) must be requested from HuggingFace:
    https://huggingface.co/facebook/sam3
  Place the downloaded sam3.pt in the working directory (or use --sam3-weights).
- If SAM3 raises "TypeError: 'SimpleTokenizer' object is not callable":
    pip uninstall clip -y
    pip install git+https://github.com/ultralytics/CLIP.git
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Supported file extensions ────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Default text prompts for traversable-road concept ───────────────────────────
DEFAULT_PROMPTS = [
    "traversable road",
    "dirt road",
    "off-road trail",
    "drivable terrain",
    "gravel path",
]

# ── Visualisation constants ──────────────────────────────────────────────────────
MASK_BGR    = (34, 139, 34)   # forest-green road overlay
CONTOUR_BGR = (0, 255, 255)   # cyan boundary
MASK_ALPHA  = 0.40            # overlay transparency (0 = invisible, 1 = opaque)
HUD_ALPHA   = 0.55            # HUD background transparency


# ══════════════════════════════════════════════════════════════════════════════════
#   Utility helpers
# ══════════════════════════════════════════════════════════════════════════════════

def collect_inputs(input_path: str):
    """
    Return (image_paths, video_paths, label_dir) for the given file or directory.

    Handles two layouts:
      flat/          →  test_data/orfd.png  (no labels)
      structured/    →  test_data/raw/orfd.png  +  test_data/labeled/orfd_labeled.png
    """
    p = Path(input_path)
    if p.is_file():
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            return [p], [], None
        if ext in VIDEO_EXTS:
            return [], [p], None
        sys.exit(f"[ERROR] Unsupported file type: {ext}")
    if p.is_dir():
        # Prefer structured layout: raw/ + labeled/
        raw_dir   = p / "raw"
        label_dir = p / "labeled"
        src_dir   = raw_dir if raw_dir.is_dir() else p
        ldir      = label_dir if label_dir.is_dir() else None
        imgs = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        vids = sorted(f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        return imgs, vids, ldir
    sys.exit(f"[ERROR] Path not found: {input_path}")


def find_label(img_path: Path, label_dir: Path):
    """Return the label path for an image, or None if not found.
    Convention: raw/orfd.png → labeled/orfd_labeled.png
    """
    candidate = label_dir / f"{img_path.stem}_labeled{img_path.suffix}"
    return candidate if candidate.exists() else None


def load_gt_mask(label_path: Path):
    """
    Load a 3-class label image and return binary masks.

    Label pixel values:
      255 (white) = traversable road  → positive class
        0 (black) = non-road          → negative class
      128 (gray)  = void / unknown    → excluded from all metrics

    Returns
    -------
    gt_mask   : np.ndarray [H, W] bool   True = road
    void_mask : np.ndarray [H, W] bool   True = ignore this pixel
    """
    label = cv2.imread(str(label_path))
    if label is None:
        return None, None
    # Explicit BGR→Gray conversion avoids IMREAD_GRAYSCALE returning 3D on some builds
    gray = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY) if label.ndim == 3 else label
    gt_mask   = gray > 200                       # white pixels → road
    void_mask = (gray > 50) & (gray < 200)       # gray pixels  → void
    return gt_mask, void_mask


def compute_gt_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray, void_mask: np.ndarray) -> dict:
    """
    Compute segmentation accuracy against ground truth, excluding void pixels.

    Metrics
    -------
    iou            Intersection over Union  = TP / (TP + FP + FN)
    precision      TP / (TP + FP)
    recall         TP / (TP + FN)
    f1             2 * precision * recall / (precision + recall)
    pixel_accuracy (TP + TN) / all valid pixels
    """
    # Ensure all inputs are strictly 2-D (guard against 3-D grayscale arrays)
    pred_mask = np.squeeze(pred_mask)
    gt_mask   = np.squeeze(gt_mask)
    void_mask = np.squeeze(void_mask)

    valid  = ~void_mask                    # boolean mask of pixels that count
    pred_v = pred_mask[valid]
    gt_v   = gt_mask[valid]

    TP = int(( pred_v &  gt_v).sum())
    TN = int((~pred_v & ~gt_v).sum())
    FP = int(( pred_v & ~gt_v).sum())
    FN = int((~pred_v &  gt_v).sum())

    iou        = TP / (TP + FP + FN)          if (TP + FP + FN) > 0         else 0.0
    precision  = TP / (TP + FP)               if (TP + FP)       > 0         else 0.0
    recall     = TP / (TP + FN)               if (TP + FN)        > 0         else 0.0
    f1         = 2 * precision * recall / (precision + recall) \
                                               if (precision + recall) > 0     else 0.0
    pixel_acc  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "iou":            round(iou,       4),
        "precision":      round(precision, 4),
        "recall":         round(recall,    4),
        "f1":             round(f1,        4),
        "pixel_accuracy": round(pixel_acc, 4),
    }


def extract_masks(result, h: int, w: int):
    """
    Extract and combine all segmentation masks from an Ultralytics result object.

    Returns
    -------
    combined : np.ndarray [H, W] bool
        Union of all detected road instance masks.
    confs : list[float]
        Per-instance confidence scores.
    """
    combined = np.zeros((h, w), dtype=bool)
    confs = []

    if result is None or result.masks is None:
        return combined, confs

    for i, m in enumerate(result.masks.data):
        m_np = m.cpu().numpy().astype(np.uint8)
        if m_np.shape != (h, w):
            m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
        combined |= m_np.astype(bool)

        # Confidence: prefer boxes.conf, then masks.conf, fallback to 1.0
        conf = 1.0
        if result.boxes is not None and i < len(result.boxes.conf):
            conf = float(result.boxes.conf[i].cpu())
        elif (hasattr(result.masks, "conf")
              and result.masks.conf is not None
              and i < len(result.masks.conf)):
            conf = float(result.masks.conf[i].cpu())
        confs.append(conf)

    return combined, confs


def overlay_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply semi-transparent green overlay and cyan contour boundary."""
    out = frame.copy()
    if not mask.any():
        return out
    color_layer = frame.copy()
    color_layer[mask] = MASK_BGR
    out = cv2.addWeighted(color_layer, MASK_ALPHA, out, 1.0 - MASK_ALPHA, 0)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, CONTOUR_BGR, 2)
    return out


def draw_hud(frame: np.ndarray, metrics: dict, model_label: str) -> np.ndarray:
    """Render a semi-transparent HUD with per-frame metrics in the top-left corner."""
    lines = [
        f"Model : {model_label}",
        f"Infer : {metrics['inference_time_ms']:.0f} ms   FPS: {metrics['fps']:.1f}",
        f"Cover : {metrics['road_coverage_pct']:.1f}%   Det: {metrics['num_detections']}",
        f"Conf  : {metrics['mean_confidence']:.3f}",
    ]
    if "temporal_iou" in metrics:
        lines.append(f"T-IoU : {metrics['temporal_iou']:.3f}")
    # Ground-truth metrics (images only)
    if "iou" in metrics:
        lines.append(f"IoU   : {metrics['iou']:.3f}   F1: {metrics['f1']:.3f}")
        lines.append(f"Prec  : {metrics['precision']:.3f}   Rec: {metrics['recall']:.3f}")

    pad, line_h = 8, 24
    rect_h = len(lines) * line_h + pad * 2
    rect_w = 370

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (rect_w, rect_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, HUD_ALPHA, frame, 1.0 - HUD_ALPHA, 0)

    for i, ln in enumerate(lines):
        y = pad + (i + 1) * line_h - 4
        cv2.putText(
            frame, ln, (pad, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 1, cv2.LINE_AA,
        )
    return frame


def compute_metrics(
    mask: np.ndarray,
    confs: list,
    t_ms: float,
    prev_mask=None,
) -> dict:
    """
    Compute per-frame / per-image metrics.

    Metrics
    -------
    inference_time_ms : float   Wall-clock inference time in milliseconds.
    fps               : float   Effective frames per second from inference time.
    road_coverage_pct : float   Fraction of image covered by road mask (%).
    mean_confidence   : float   Mean confidence across all detected instances.
    num_detections    : int     Number of distinct road instances found.
    temporal_iou      : float   IoU with the previous frame's mask (video only).
    """
    h, w = mask.shape
    road_px = int(mask.sum())
    m = {
        "inference_time_ms": round(t_ms, 2),
        "fps":               round(1000.0 / t_ms, 1) if t_ms > 0 else 0.0,
        "road_coverage_pct": round(100.0 * road_px / (h * w), 2),
        "mean_confidence":   round(float(np.mean(confs)), 4) if confs else 0.0,
        "num_detections":    len(confs),
    }
    if prev_mask is not None:
        inter = int((mask & prev_mask).sum())
        union = int((mask | prev_mask).sum())
        m["temporal_iou"] = round(inter / union, 4) if union > 0 else 1.0
    return m


def print_summary(all_m: list, model_name: str, label: str):
    """Print a formatted aggregate-metrics table to stdout."""
    if not all_m:
        return

    col_w = 30
    sep  = "=" * 66
    dash = "-" * 66

    def _section(title, keys):
        vals_exist = any(k in all_m[0] for k in keys)
        if not vals_exist:
            return
        print(f"  -- {title} --")
        for k in keys:
            vals = [m[k] for m in all_m if k in m]
            if vals:
                print(f"  {k:{col_w}}  {np.mean(vals):>8.3f}  {np.min(vals):>8.3f}  {np.max(vals):>8.3f}")

    print(f"\n{sep}")
    print(f"  Results: {model_name.upper():<8}  |  {label}")
    print(dash)
    print(f"  {'Metric':{col_w}}  {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print(dash)

    _section("Performance", ["inference_time_ms", "fps"])
    _section("Model output", ["road_coverage_pct", "mean_confidence", "num_detections"])
    _section("Video consistency", ["temporal_iou"])
    _section("Ground truth", ["iou", "f1", "precision", "recall", "pixel_accuracy"])

    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════════
#   Model wrappers
# ══════════════════════════════════════════════════════════════════════════════════

class YOLO26Segmentor:
    """
    YOLOE-26 open-vocabulary segmentor.

    Uses Ultralytics YOLOE-26-seg with text prompts, so it works on arbitrary
    off-road road concepts without any fine-tuning.
    """

    def __init__(self, size: str = "l", conf: float = 0.25, prompts: list = None):
        from ultralytics import YOLO

        name = f"yoloe-26{size}-seg.pt"
        print(f"[YOLO26] Loading {name} (auto-download if not cached) ...")
        self.model   = YOLO(name)
        self.conf    = conf
        self.prompts = prompts or DEFAULT_PROMPTS
        self.model.set_classes(self.prompts)
        print(f"[YOLO26] Text classes : {self.prompts}")

    def infer(self, source) -> tuple:
        """Run inference on a file path or numpy BGR frame."""
        t0  = time.perf_counter()
        res = self.model.predict(source, conf=self.conf, verbose=False)
        t_ms = (time.perf_counter() - t0) * 1000
        return res[0], t_ms


class SAM3Segmentor:
    """
    SAM3 semantic predictor for concept segmentation.

    Uses SAM3SemanticPredictor for images (text prompt) and
    SAM3VideoSemanticPredictor for videos (temporal tracking).
    """

    def __init__(self, weights: str = "sam3.pt", conf: float = 0.25, prompts: list = None):
        weights_path = Path(weights)
        if not weights_path.exists():
            sys.exit(
                f"\n[ERROR] SAM3 weights not found: {weights}\n"
                "  1. Request access at: https://huggingface.co/facebook/sam3\n"
                "  2. Download sam3.pt and place it in the working directory.\n"
                "  3. Or pass the path via --sam3-weights /path/to/sam3.pt\n"
            )

        from ultralytics.models.sam import SAM3SemanticPredictor

        print(f"[SAM3] Loading {weights} ...")
        self._ov = dict(
            conf=conf, task="segment", mode="predict",
            model=str(weights_path), verbose=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=self._ov)
        self.prompts   = prompts or DEFAULT_PROMPTS
        self.weights   = str(weights_path)
        self.conf      = conf
        print(f"[SAM3] Text prompts : {self.prompts}")

    def infer(self, source) -> tuple:
        """Run image inference. source = file path string or numpy BGR frame."""
        t0 = time.perf_counter()
        self.predictor.set_image(source)
        res  = self.predictor(text=self.prompts)
        t_ms = (time.perf_counter() - t0) * 1000
        return (res[0] if res else None), t_ms



# ══════════════════════════════════════════════════════════════════════════════════
#   Per-file processing
# ══════════════════════════════════════════════════════════════════════════════════

def run_image(seg, img_path: Path, out_dir: Path, model_name: str, label_dir=None):
    """Segment a single image. Saves annotated output, returns metrics dict."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read: {img_path.name}")
        return None

    result, t_ms = seg.infer(str(img_path))
    h, w = img.shape[:2]
    mask, confs = extract_masks(result, h, w)

    m = compute_metrics(mask, confs, t_ms)
    m["file"] = img_path.name

    # ── Ground-truth metrics (when labels are available) ─────────────────────
    if label_dir is not None:
        lp = find_label(img_path, label_dir)
        if lp:
            gt_mask, void_mask = load_gt_mask(lp)
            if gt_mask is not None:
                m.update(compute_gt_metrics(mask, gt_mask, void_mask))
        else:
            print(f"  [WARN] No label found for {img_path.name} in {label_dir}")

    annotated = overlay_mask(img, mask)
    annotated = draw_hud(annotated, m, model_name.upper())

    out = out_dir / f"{img_path.stem}_{model_name}_road{img_path.suffix}"
    cv2.imwrite(str(out), annotated)

    gt_str = (f"  IoU={m['iou']:.3f}  F1={m['f1']:.3f}"
               if "iou" in m else "  (no label)")
    print(
        f"  {img_path.name:30s}  "
        f"cov={m['road_coverage_pct']:5.1f}%  "
        f"t={t_ms:.0f}ms"
        f"{gt_str}  → {out.name}"
    )
    return m


def run_video(seg, vid_path: Path, out_dir: Path, model_name: str) -> list:
    """
    Segment a video. Saves annotated MP4 output.
    Returns a list of per-frame metrics dicts.
    """
    # ── Open video to read properties ────────────────────────────────────────────
    probe = cv2.VideoCapture(str(vid_path))
    if not probe.isOpened():
        print(f"  [WARN] Cannot open: {vid_path.name}")
        return []
    src_fps  = probe.get(cv2.CAP_PROP_FPS) or 25.0
    W        = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    out_path = out_dir / f"{vid_path.stem}_{model_name}_road.mp4"
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (W, H)
    )
    print(f"\n  [{model_name.upper()}] {vid_path.name}  ({W}×{H}  {src_fps:.1f} fps  {n_frames} frames)")

    all_m, prev_mask, fi = [], None, 0

    # Both models use frame-by-frame inference.
    # SAM3VideoSemanticPredictor loads all frames at once and stalls on long
    # high-resolution videos with limited VRAM, so we use seg.infer() per frame
    # for both backends — consistent, memory-safe, and easier to track progress.
    cap = cv2.VideoCapture(str(vid_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result, t_ms = seg.infer(frame)
        mask, confs  = extract_masks(result, H, W)
        m = compute_metrics(mask, confs, t_ms, prev_mask)
        m["frame"] = fi

        annotated = overlay_mask(frame, mask)
        annotated = draw_hud(annotated, m, model_name.upper())
        writer.write(annotated)
        all_m.append(m)
        prev_mask = mask
        fi += 1

        if fi % 30 == 0:
            print(
                f"    frame {fi:4d}/{n_frames}  "
                f"cov={m['road_coverage_pct']:.1f}%  "
                f"conf={m['mean_confidence']:.3f}  "
                f"t={t_ms:.0f}ms"
            )
    cap.release()

    writer.release()
    print(f"  Saved: {out_path.name}  ({fi} frames processed)")
    return all_m


# ══════════════════════════════════════════════════════════════════════════════════
#   CLI
# ══════════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Off-road traversable road segmentation — SAM3 vs YOLO26",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", required=True, choices=["sam3", "yolo26"],
        help="Segmentation backend: 'sam3' or 'yolo26'",
    )
    p.add_argument(
        "--input", default="test_data",
        help="Image/video file or directory to process  [default: test_data/]",
    )
    p.add_argument(
        "--output", default="output",
        help="Directory for annotated outputs           [default: output/]",
    )
    p.add_argument(
        "--conf", type=float, default=0.25,
        help="Detection confidence threshold            [default: 0.25]",
    )
    p.add_argument(
        "--prompts", nargs="+", default=None,
        help="Custom text prompts for the road concept  [default: built-in set]",
    )
    p.add_argument(
        "--model-size", default="l", choices=["n", "s", "m", "l", "x"],
        help="YOLOE-26 model size variant               [default: l, ignored for sam3]",
    )
    p.add_argument(
        "--sam3-weights", default="sam3.pt",
        help="Path to sam3.pt weights file              [default: sam3.pt]",
    )
    p.add_argument(
        "--report", action="store_true",
        help="Save a metrics_<model>.json report to the output directory",
    )
    return p


def main():
    args = build_parser().parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, videos, label_dir = collect_inputs(args.input)
    total = len(images) + len(videos)

    hdr = "-" * 66
    print(f"\n{hdr}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Inputs  : {len(images)} image(s), {len(videos)} video(s)  [{args.input}]")
    print(f"  Labels  : {label_dir or 'none found'}")
    print(f"  Output  : {out_dir.resolve()}")
    print(f"  Prompts : {args.prompts or DEFAULT_PROMPTS}")
    print(hdr)

    if total == 0:
        sys.exit("[ERROR] No supported image/video files found in the input path.")

    # ── Load model ───────────────────────────────────────────────────────────────
    if args.model == "yolo26":
        seg = YOLO26Segmentor(
            size=args.model_size, conf=args.conf, prompts=args.prompts
        )
    else:
        seg = SAM3Segmentor(
            weights=args.sam3_weights, conf=args.conf, prompts=args.prompts
        )

    report = {"model": args.model, "images": [], "videos": {}}

    # ── Process images ───────────────────────────────────────────────────────────
    if images:
        print(f"\n[Images]")
        img_metrics = []
        for p in images:
            m = run_image(seg, p, out_dir, args.model, label_dir)
            if m:
                img_metrics.append(m)
        report["images"] = img_metrics
        if img_metrics:
            print_summary(img_metrics, args.model, f"{len(img_metrics)} image(s)")

    # ── Process videos ───────────────────────────────────────────────────────────
    if videos:
        print(f"\n[Videos]")
        for p in videos:
            vm = run_video(seg, p, out_dir, args.model)
            report["videos"][p.name] = vm
            if vm:
                print_summary(vm, args.model, p.name)

    # ── Save JSON report ─────────────────────────────────────────────────────────
    if args.report:
        rp = out_dir / f"metrics_{args.model}.json"
        with open(rp, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON metrics report → {rp}")

    print(f"\nAll outputs saved to: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
