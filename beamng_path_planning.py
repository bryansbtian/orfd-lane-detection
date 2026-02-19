"""
BeamNG.tech Path Planning Script for Off-Road Autonomous Driving

Uses a trained traversability segmentation model to extract a drivable path
from camera images and autonomously steer the vehicle along it.

Features:
  - Real-time path extraction from traversability mask
  - Image-space pure-pursuit path following controller
  - Safe-fail: controlled stop when no traversable surface (SR-1)
  - Safe-fail: controlled stop when obstacle detected ahead (SR-2)
  - Speed limited to configurable max (default 20 mph / PR-1)
  - Planned path + road boundary overlay at >20 FPS (PR-2)

Usage:
    python beamng_path_planning.py \
        --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth \
        --arch deeplabv3p --encoder resnet50

Controls (in the OpenCV window):
    P       - Toggle autonomous / manual mode
    W/S     - Throttle / Brake  (manual mode only)
    A/D     - Steer left / right (manual mode only)
    SPACE   - Handbrake
    T       - Toggle traversability overlay
    O       - Toggle path overlay
    R       - Toggle recording
    [/]     - Decrease / increase max speed (1 mph steps)
    G       - Print current position
    Q/ESC   - Quit
"""

import argparse
import time
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import cv2

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise RuntimeError("segmentation-models-pytorch is required: pip install segmentation-models-pytorch")

try:
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors.camera import Camera
except ImportError:
    raise RuntimeError("beamngpy is required: pip install beamngpy")


# ── Model helpers (reused from beamng_inference.py) ───────────────────────────

ARCH_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "unetpp": "UnetPlusPlus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "deeplabv3": "DeepLabV3",
    "deeplabv3p": "DeepLabV3Plus",
    "deeplabv3plus": "DeepLabV3Plus",
}

OFFROAD_MAPS = {
    "jungle_rock_island": {
        "spawns": [
            ((122.03, -74.66, 158.65), (0, 0, 0.966, 0.259)),
            ((353.80, -151.3, 160.19), (0, 0, -0.875, 0.484)),
        ],
    },
    "johnson_valley": {
        "spawns": [
            ((-1152, 291.05, 119.84), (0, 0, 0, 1)),
        ],
    },
    "west_coast_usa": {
        "spawns": [
            ((565.13, 27.664, 148.06), (0, 0, -0.934, 0.355)),
        ],
    },
    "utah": {
        "spawns": [
            ((575, 118, 132), (0, 0, 0, 1)),
            ((200, 200, 135), (0, 0, 0.707, 0.707)),
        ],
    },
    "small_island": {
        "spawns": [
            ((240, 215, 37), (0, 0, 0, 1)),
            ((145, 240, 37), (0, 0, 0.707, 0.707)),
        ],
    },
    "east_coast_usa": {
        "spawns": [
            ((-700, 50, 35), (0, 0, 0, 1)),
        ],
    },
    "Italy": {
        "spawns": [
            ((200, 200, 200), (0, 0, 0, 1)),
        ],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="BeamNG.tech off-road path planning")

    # BeamNG
    p.add_argument("--beamng-home", type=str,
                   default=r"E:\BeamNG.tech.v0.38.3.0",
                   help="Path to BeamNG.tech installation")
    p.add_argument("--beamng-user", type=str, default=None,
                   help="Path for BeamNG user folder (default: auto)")
    p.add_argument("--map", type=str, default="jungle_rock_island",
                   help="Map to load")
    p.add_argument("--spawn-idx", type=int, default=0,
                   help="Spawn point index")
    p.add_argument("--vehicle", type=str, default="pickup",
                   help="Vehicle model")

    # Camera
    p.add_argument("--width", type=int, default=1280, help="Camera width")
    p.add_argument("--height", type=int, default=720, help="Camera height")
    p.add_argument("--fov", type=float, default=120, help="Camera vertical FOV")
    p.add_argument("--fps", type=float, default=30.0, help="Target FPS")

    # Model
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pth model checkpoint")
    p.add_argument("--arch", type=str, default="deeplabv3p",
                   help="Model architecture")
    p.add_argument("--encoder", type=str, default="resnet50",
                   help="Encoder backbone")
    p.add_argument("--img-size", type=int, default=256,
                   help="Model input size")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Sigmoid threshold for binary mask")

    # Output
    p.add_argument("--output-dir", type=str, default="beamng_recordings",
                   help="Directory for recorded videos")
    p.add_argument("--alpha", type=float, default=0.45,
                   help="Overlay transparency")

    # Path planning
    p.add_argument("--mode", type=str, default="autonomous",
                   choices=["manual", "autonomous"],
                   help="Control mode")
    p.add_argument("--lookahead-ratio", type=float, default=0.4,
                   help="Lookahead distance as ratio of ROI height")
    p.add_argument("--steering-gain", type=float, default=2.5,
                   help="Proportional gain for steering")
    p.add_argument("--max-speed", type=float, default=20.0,
                   help="Maximum speed in mph")
    p.add_argument("--safe-fail-threshold", type=float, default=0.05,
                   help="Min traversable area ratio before safe-fail")
    p.add_argument("--obstacle-threshold", type=float, default=0.80,
                   help="Non-traversable fraction in lookahead zone to trigger stop")

    return p.parse_args()


# ── Model functions (from beamng_inference.py) ────────────────────────────────

def build_model(arch, encoder, encoder_weights="imagenet"):
    arch_key = arch.strip().lower()
    if arch_key not in ARCH_ALIASES:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {sorted(ARCH_ALIASES.keys())}")
    cls_name = ARCH_ALIASES[arch_key]
    ModelCls = getattr(smp, cls_name)
    return ModelCls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=1,
        activation=None,
    )


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state_dict = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
        if state_dict is None and all(hasattr(v, "shape") for v in ckpt.values()):
            state_dict = ckpt
    else:
        state_dict = ckpt
    if state_dict is None:
        raise RuntimeError(f"Could not find state_dict in checkpoint: {ckpt_path}")
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")


def preprocess(bgr, input_size, device):
    resized = cv2.resize(bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    x = torch.from_numpy(rgb).to(device=device, dtype=dtype) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 1, 3)
    x = (x - mean) / std
    x = x.permute(2, 0, 1).unsqueeze(0)
    return x


@torch.no_grad()
def predict_mask(model, x, threshold):
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        y = y.get("out", next(iter(y.values())))
    prob = torch.sigmoid(y)[0, 0]
    mask = (prob >= threshold).to(torch.uint8) * 255
    return mask.cpu().numpy()


def overlay_green(bgr, mask_255, alpha):
    if mask_255.shape[:2] != bgr.shape[:2]:
        mask_255 = cv2.resize(mask_255, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    out = bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    m = (mask_255 > 0)[:, :, None]
    out = np.where(m, (alpha * green + (1 - alpha) * out).astype(np.uint8), out)
    return out


# ── Path Planner ──────────────────────────────────────────────────────────────

class PathPlanner:
    """Extracts a drivable centerline path from a binary traversability mask."""

    def __init__(self, img_height, img_width,
                 roi_top_pct=0.30, roi_bottom_pct=1.0,
                 sample_step=5, smooth_window=7):
        self.img_h = img_height
        self.img_w = img_width
        self.roi_top = int(img_height * roi_top_pct)
        self.roi_bot = int(img_height * roi_bottom_pct)
        self.sample_step = sample_step
        self.smooth_window = smooth_window

    def extract_path(self, mask):
        """Return (waypoints, left_boundary, right_boundary, metrics).

        waypoints : list of (x, y) in full-image coords — the centerline
        left_boundary / right_boundary : lists of (x, y) for corridor edges
        metrics   : dict with traversable_pct, avg_corridor_width, min_corridor_width
        """
        roi = mask[self.roi_top:self.roi_bot, :]
        roi_h, roi_w = roi.shape

        centers = []
        lefts = []
        rights = []
        widths = []

        for r in range(0, roi_h, self.sample_step):
            row = roi[r, :]
            trav = np.where(row > 127)[0]
            if len(trav) == 0:
                continue
            left = int(trav[0])
            right = int(trav[-1])
            center = (left + right) // 2
            width = right - left
            y = self.roi_top + r  # map back to full-image coords
            centers.append((center, y))
            lefts.append((left, y))
            rights.append((right, y))
            widths.append(width)

        # Smooth centerline
        if len(centers) >= self.smooth_window:
            xs = np.array([c[0] for c in centers], dtype=np.float64)
            kernel = np.ones(self.smooth_window) / self.smooth_window
            xs_smooth = np.convolve(xs, kernel, mode="same")
            centers = [(int(xs_smooth[i]), centers[i][1]) for i in range(len(centers))]

        # Metrics
        total_roi_pixels = roi_h * roi_w
        trav_pixels = int(np.sum(roi > 127))
        traversable_pct = trav_pixels / total_roi_pixels if total_roi_pixels > 0 else 0.0
        avg_w = float(np.mean(widths)) if widths else 0.0
        min_w = float(np.min(widths)) if widths else 0.0

        metrics = {
            "traversable_pct": traversable_pct,
            "avg_corridor_width": avg_w,
            "min_corridor_width": min_w,
        }
        return centers, lefts, rights, metrics

    def detect_obstacle(self, mask, obstacle_threshold, centers=None):
        """Check if the path ahead is blocked.

        Instead of a fixed center zone, check a corridor around the planned
        centerline in the upper (farther) portion of the ROI.  Falls back to
        the image centre when no path waypoints are available.

        Returns True when the non-traversable fraction along the path corridor
        exceeds *obstacle_threshold* (default 0.80 = nearly fully blocked).
        """
        roi = mask[self.roi_top:self.roi_bot, :]
        roi_h, roi_w = roi.shape

        # Only check the top half of the ROI (further ahead)
        zone_bot = roi_h // 2

        # Build a corridor ±corridor_half pixels around each centerline point
        corridor_half = roi_w // 6  # ~17% of width on each side
        total_pixels = 0
        non_trav_pixels = 0

        if centers:
            for (cx, cy) in centers:
                row_in_roi = cy - self.roi_top
                if row_in_roi < 0 or row_in_roi >= zone_bot:
                    continue
                left = max(0, cx - corridor_half)
                right = min(roi_w, cx + corridor_half)
                segment = roi[row_in_roi, left:right]
                total_pixels += len(segment)
                non_trav_pixels += int(np.sum(segment <= 127))
        else:
            # Fallback: center strip
            left = roi_w // 4
            right = roi_w * 3 // 4
            zone = roi[:zone_bot, left:right]
            total_pixels = zone.size
            non_trav_pixels = int(np.sum(zone <= 127))

        if total_pixels == 0:
            return False
        return (non_trav_pixels / total_pixels) > obstacle_threshold


# ── Path Following Controller ─────────────────────────────────────────────────

class PathFollowingController:
    """Image-space pure-pursuit controller that converts a planned path into
    steering / throttle / brake commands."""

    MPS_TO_MPH = 2.23694

    def __init__(self, img_width, img_height,
                 lookahead_ratio=0.4,
                 steering_kp=2.5,
                 max_steering_delta=0.15,
                 max_speed_mph=20.0,
                 base_throttle=0.35,
                 roi_top_pct=0.30):
        self.img_w = img_width
        self.img_h = img_height
        self.img_cx = img_width // 2
        self.roi_top = int(img_height * roi_top_pct)
        self.roi_bot = img_height

        # Lookahead row in full-image coords
        roi_h = self.roi_bot - self.roi_top
        self.lookahead_row = self.roi_top + int(roi_h * (1.0 - lookahead_ratio))

        self.steering_kp = steering_kp
        self.max_steering_delta = max_steering_delta
        self.max_speed_mps = max_speed_mph / self.MPS_TO_MPH
        self.base_throttle = base_throttle

        # State
        self.prev_steering = 0.0

    def compute(self, waypoints, metrics, current_speed_mps):
        """Return (steering, throttle, brake, lookahead_pt).

        lookahead_pt is the (x, y) used for visualization (or None).
        """
        if not waypoints:
            return 0.0, 0.0, 0.5, None

        # Find the waypoint closest to the lookahead row
        lookahead_pt = self._find_lookahead(waypoints)
        if lookahead_pt is None:
            return 0.0, 0.0, 0.5, None

        # Lateral error normalised to [-1, 1]
        lateral_err = (lookahead_pt[0] - self.img_cx) / (self.img_w / 2)

        # Proportional steering
        steer_raw = self.steering_kp * lateral_err
        steer_raw = np.clip(steer_raw, -1.0, 1.0)

        # Rate-limit
        delta = steer_raw - self.prev_steering
        if abs(delta) > self.max_steering_delta:
            delta = np.sign(delta) * self.max_steering_delta
        steering = np.clip(self.prev_steering + delta, -1.0, 1.0)
        self.prev_steering = steering

        # Throttle / brake — speed control
        speed_mph = current_speed_mps * self.MPS_TO_MPH
        speed_err = self.max_speed_mps - current_speed_mps

        if speed_err < -0.5:
            # Over speed limit
            throttle = 0.0
            brake = 0.3
        else:
            # Scale throttle: reduce when corridor is narrow or steering is large
            corridor_factor = min(metrics["avg_corridor_width"] / 200.0, 1.0)
            steer_factor = 1.0 - 0.5 * abs(steering)
            throttle = self.base_throttle * corridor_factor * steer_factor
            # Ramp down near speed limit
            if speed_err < 2.0 and speed_err >= 0:
                throttle *= speed_err / 2.0
            throttle = np.clip(throttle, 0.0, 0.6)
            brake = 0.0

        return float(steering), float(throttle), float(brake), lookahead_pt

    def _find_lookahead(self, waypoints):
        best = None
        best_dist = float("inf")
        for (x, y) in waypoints:
            d = abs(y - self.lookahead_row)
            if d < best_dist:
                best_dist = d
                best = (x, y)
        return best

    def reset(self):
        self.prev_steering = 0.0


# ── Visualization helpers ─────────────────────────────────────────────────────

def draw_path_overlay(display, centers, lefts, rights, lookahead_pt):
    """Draw planned path, corridor boundaries, and lookahead point."""
    # Left boundary (red)
    if len(lefts) > 1:
        pts = np.array(lefts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [pts], False, (0, 0, 255), 2)

    # Right boundary (red)
    if len(rights) > 1:
        pts = np.array(rights, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [pts], False, (0, 0, 255), 2)

    # Centerline (yellow)
    if len(centers) > 1:
        pts = np.array(centers, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(display, [pts], False, (0, 255, 255), 3)

    # Lookahead target (orange circle)
    if lookahead_pt is not None:
        cv2.circle(display, (int(lookahead_pt[0]), int(lookahead_pt[1])),
                   12, (0, 165, 255), -1)
        cv2.circle(display, (int(lookahead_pt[0]), int(lookahead_pt[1])),
                   12, (0, 0, 0), 2)


def draw_hud(display, current_fps, speed_mph, max_speed_mph,
             metrics, autonomous, safe_fail_reason, recording,
             record_start, show_overlay, show_path):
    h, w = display.shape[:2]

    # Top bar
    mode_str = "AUTO" if autonomous else "MANUAL"
    overlay_str = "ON" if show_overlay else "OFF"
    path_str = "ON" if show_path else "OFF"
    rec_str = ""
    if recording and record_start:
        rec_str = f"  REC {time.time() - record_start:.1f}s"

    line1 = (f"FPS: {current_fps:.0f}  Speed: {speed_mph:.1f}/{max_speed_mph:.0f} mph  "
             f"Mode: {mode_str}  Overlay: {overlay_str}  Path: {path_str}{rec_str}")
    cv2.putText(display, line1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
    cv2.putText(display, line1, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Traversability bar
    trav_pct = metrics.get("traversable_pct", 0) * 100
    corr_w = metrics.get("avg_corridor_width", 0)
    line2 = f"Traversable: {trav_pct:.0f}%  Corridor: {corr_w:.0f}px"
    cv2.putText(display, line2, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(display, line2, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

    # Safe-fail banner
    if safe_fail_reason:
        banner = f"STOP - {safe_fail_reason}"
        text_size = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        bx = (w - text_size[0]) // 2
        by = 110
        cv2.rectangle(display, (bx - 15, by - 35), (bx + text_size[0] + 15, by + 10),
                      (0, 0, 180), -1)
        cv2.putText(display, banner, (bx, by),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    # Recording indicator
    if recording:
        cv2.circle(display, (w - 25, 25), 10, (0, 0, 255), -1)

    # Controls reminder
    controls = ("P=Auto/Manual  W/S=Throttle  A/D=Steer  T=Overlay  "
                "O=Path  [/]=Speed  R=Rec  Q=Quit")
    cv2.putText(display, controls, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path.resolve()}")
        sys.exit(1)

    # Build and load model
    print(f"[INFO] Loading model: arch={args.arch}, encoder={args.encoder}, device={device}")
    model = build_model(args.arch, args.encoder)
    model.to(device)
    load_checkpoint(model, ckpt_path, device)
    model.eval()
    if device.type == "cuda":
        model.half()
        torch.backends.cudnn.benchmark = True
    print("[INFO] Model loaded successfully.")

    # ── BeamNG setup ──────────────────────────────────────────────────────────
    print(f"[INFO] Connecting to BeamNG.tech at {args.beamng_home}")
    bng_kwargs = dict(host="localhost", port=64256, home=args.beamng_home)
    if args.beamng_user:
        bng_kwargs["user"] = args.beamng_user

    bng = BeamNGpy(**bng_kwargs)
    bng.open(launch=True)

    video_writer = None
    camera = None

    try:
        map_name = args.map
        print(f"[INFO] Loading map: {map_name}")
        scenario = Scenario(map_name, "offroad_pathplanning")

        map_cfg = OFFROAD_MAPS.get(map_name)
        if map_cfg:
            spawn_idx = min(args.spawn_idx, len(map_cfg["spawns"]) - 1)
            spawn_pos, spawn_rot = map_cfg["spawns"][spawn_idx]
            print(f"[INFO] Spawning at predefined position: {spawn_pos}")
        else:
            spawn_pos, spawn_rot = (0, 0, 0), (0, 0, 0, 1)

        vehicle = Vehicle("ego", model=args.vehicle, license="OFFROAD")
        scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot, cling=True)

        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()
        vehicle.connect(bng)

        # Attach front-facing camera
        camera = Camera(
            name="front_cam",
            bng=bng,
            vehicle=vehicle,
            pos=(0, -2.5, 0.8),
            dir=(0, -1, -0.1),
            up=(0, 0, 1),
            resolution=(args.width, args.height),
            field_of_view_y=args.fov,
            near_far_planes=(0.1, 500.0),
            requested_update_time=0.01,
            update_priority=1.0,
            is_render_colours=True,
            is_render_annotations=False,
            is_render_depth=False,
            is_using_shared_memory=True,
            is_streaming=True,
        )
        print(f"[INFO] Camera attached: {args.width}x{args.height} @ {args.fov} FOV")

        # ── Path planner & controller ─────────────────────────────────────────
        planner = PathPlanner(args.height, args.width)
        controller = PathFollowingController(
            args.width, args.height,
            lookahead_ratio=args.lookahead_ratio,
            steering_kp=args.steering_gain,
            max_speed_mph=args.max_speed,
        )

        # ── State ─────────────────────────────────────────────────────────────
        autonomous = (args.mode == "autonomous")
        show_overlay = True
        show_path = True
        recording = False
        video_writer = None
        record_start = None
        output_dir = Path(args.output_dir)
        max_speed_mph = args.max_speed

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        # Manual control state
        throttle = 0.0
        brake = 0.0
        steering = 0.0

        # Safe-fail state
        safe_fail_frames = 0       # consecutive frames with no traversable surface
        SAFE_FAIL_PATIENCE = 3     # require N consecutive frames before stopping
        safe_fail_reason = None

        cv2.namedWindow("BeamNG Path Planning", cv2.WINDOW_NORMAL)

        print("\n" + "=" * 60)
        print("  BeamNG Off-Road Path Planning")
        print("=" * 60)
        print(f"  Mode: {'AUTONOMOUS' if autonomous else 'MANUAL'}")
        print(f"  Max speed: {max_speed_mph:.0f} mph")
        print("  P = Toggle Auto/Manual   T = Overlay   O = Path")
        print("  W/S/A/D = Drive (manual)  [/] = Speed   R = Record")
        print("  Q/ESC = Quit")
        print("=" * 60 + "\n")

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            # 1. Capture frame
            images = camera.stream()
            colour = images.get("colour")
            if colour is None:
                time.sleep(0.01)
                continue

            frame_rgb = np.array(colour)
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 2. Model inference → binary mask (always needed for path planning)
            x = preprocess(bgr, args.img_size, device)
            mask_small = predict_mask(model, x, args.threshold)
            mask = cv2.resize(mask_small, (bgr.shape[1], bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

            # 3. Path extraction
            centers, lefts, rights, metrics = planner.extract_path(mask)
            obstacle = planner.detect_obstacle(mask, args.obstacle_threshold,
                                                   centers)

            # 4. Get vehicle speed and forward/reverse detection
            vehicle.sensors.poll()
            vel = vehicle.state.get("vel", (0, 0, 0))
            speed_mps = np.linalg.norm(vel) if vel else 0.0
            speed_mph = speed_mps * PathFollowingController.MPS_TO_MPH

            # Detect if vehicle is moving backward by checking velocity
            # against the vehicle's forward direction (derived from rotation)
            rot = vehicle.state.get("rotation", None)
            moving_backward = False
            if rot and vel:
                # BeamNG quaternion (x,y,z,w) — forward is -Y in vehicle frame
                qx, qy, qz, qw = rot
                # Rotate unit -Y by quaternion to get world-space forward
                fwd_x = -2.0 * (qx * qy - qw * qz)
                fwd_y = -(1.0 - 2.0 * (qx * qx + qz * qz))
                fwd_z = -2.0 * (qy * qz + qw * qx)
                # Dot product of velocity with forward direction
                dot = vel[0] * fwd_x + vel[1] * fwd_y + vel[2] * fwd_z
                moving_backward = dot < -0.5  # >0.5 m/s backward

            # 5. Safe-fail checks
            safe_fail_reason = None
            if metrics["traversable_pct"] < args.safe_fail_threshold:
                safe_fail_frames += 1
                if safe_fail_frames >= SAFE_FAIL_PATIENCE:
                    safe_fail_reason = "No traversable surface"
            elif obstacle:
                safe_fail_frames += 1
                if safe_fail_frames >= SAFE_FAIL_PATIENCE:
                    safe_fail_reason = "Obstacle ahead"
            else:
                safe_fail_frames = 0

            # 6. Control
            lookahead_pt = None
            if autonomous:
                if moving_backward:
                    # Vehicle rolling backward — full brake to hold position
                    vehicle.control(steering=0.0, throttle=0.0, brake=1.0)
                    controller.reset()
                elif safe_fail_reason:
                    # Controlled stop — full brake to hold on slopes
                    vehicle.control(steering=0.0, throttle=0.0, brake=1.0)
                    controller.reset()
                else:
                    steer_cmd, thr_cmd, brk_cmd, lookahead_pt = controller.compute(
                        centers, metrics, speed_mps)
                    vehicle.control(steering=steer_cmd, throttle=thr_cmd, brake=brk_cmd)

            # 7. Build display
            if show_overlay:
                display = overlay_green(bgr, mask, args.alpha)
            else:
                display = bgr.copy()

            if show_path:
                draw_path_overlay(display, centers, lefts, rights, lookahead_pt)

            draw_hud(display, current_fps, speed_mph, max_speed_mph,
                     metrics, autonomous, safe_fail_reason,
                     recording, record_start, show_overlay, show_path)

            # 8. Recording
            if recording and video_writer:
                side_by_side = np.hstack([bgr, display])
                video_writer.write(side_by_side)

            # 9. FPS counter
            fps_counter += 1
            fps_elapsed = time.time() - fps_timer
            if fps_elapsed >= 1.0:
                current_fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_timer = time.time()

            # 10. Show
            cv2.imshow("BeamNG Path Planning", display)
            key = cv2.waitKey(1) & 0xFF

            # ── Keyboard ──────────────────────────────────────────────────────
            if key == ord("q") or key == 27:
                break

            elif key == ord("p"):
                autonomous = not autonomous
                if not autonomous:
                    controller.reset()
                    throttle = 0.0
                    brake = 0.0
                    steering = 0.0
                    vehicle.control(steering=0.0, throttle=0.0, brake=0.0)
                print(f"[INFO] Mode: {'AUTONOMOUS' if autonomous else 'MANUAL'}")

            elif key == ord("t"):
                show_overlay = not show_overlay
                print(f"[INFO] Overlay: {'ON' if show_overlay else 'OFF'}")

            elif key == ord("o"):
                show_path = not show_path
                print(f"[INFO] Path overlay: {'ON' if show_path else 'OFF'}")

            elif key == ord("g"):
                pos = vehicle.state.get("pos", "unknown")
                rot = vehicle.state.get("rotation", "unknown")
                print(f"[POS] Position: {pos}  Rotation: {rot}")

            elif key == ord("["):
                max_speed_mph = max(5.0, max_speed_mph - 1.0)
                controller.max_speed_mps = max_speed_mph / PathFollowingController.MPS_TO_MPH
                print(f"[INFO] Max speed: {max_speed_mph:.0f} mph")

            elif key == ord("]"):
                max_speed_mph = min(40.0, max_speed_mph + 1.0)
                controller.max_speed_mps = max_speed_mph / PathFollowingController.MPS_TO_MPH
                print(f"[INFO] Max speed: {max_speed_mph:.0f} mph")

            elif key == ord("r"):
                if not recording:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = output_dir / f"beamng_pathplan_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, args.fps,
                        (args.width * 2, args.height))
                    recording = True
                    record_start = time.time()
                    print(f"[REC] Recording started: {video_path}")
                else:
                    recording = False
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                    elapsed = time.time() - record_start if record_start else 0
                    print(f"[REC] Recording stopped ({elapsed:.1f}s)")
                    record_start = None

            # Manual driving (only when not autonomous)
            elif not autonomous:
                if key == ord("w"):
                    throttle = min(throttle + 0.15, 1.0)
                    brake = 0.0
                elif key == ord("s"):
                    brake = min(brake + 0.15, 1.0)
                    throttle = 0.0
                elif key == ord("a"):
                    steering = max(steering - 0.1, -1.0)
                elif key == ord("d"):
                    steering = min(steering + 0.1, 1.0)
                elif key == ord(" "):
                    brake = 1.0
                    throttle = 0.0

                steering *= 0.92
                if abs(steering) < 0.01:
                    steering = 0.0
                vehicle.control(steering=steering, throttle=throttle, brake=brake)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        print("[INFO] Cleaning up...")
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        try:
            camera.remove()
        except Exception:
            pass
        bng.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
