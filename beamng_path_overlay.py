"""
BeamNG.tech Path Overlay Script

Uses BeamNG's built-in AI for driving while overlaying the planned path
extracted from the traversability segmentation mask.

Usage:
    python beamng_path_overlay.py \
        --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth \
        --arch deeplabv3p --encoder resnet50

Controls (in the OpenCV window):
    W/S     - Throttle/Brake  (manual mode)
    A/D     - Steer left/right (manual mode)
    SPACE   - Handbrake
    P       - Toggle AI driving (random waypoints)
    T       - Toggle traversability overlay
    O       - Toggle path overlay
    R       - Toggle recording
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


# ── Model helpers ─────────────────────────────────────────────────────────────

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
    p = argparse.ArgumentParser(description="BeamNG.tech path overlay with AI driving")

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
    p.add_argument("--fov", type=float, default=70, help="Camera vertical FOV")
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

    # AI speed
    p.add_argument("--ai-speed", type=float, default=8.0,
                   help="AI speed limit in m/s (default 8 ~ 18 mph)")

    return p.parse_args()


# ── Model functions ───────────────────────────────────────────────────────────

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


# ── Path extraction ───────────────────────────────────────────────────────────

def extract_path(mask, roi_top_pct=0.30, sample_step=5, smooth_window=7):
    """Extract centerline, left/right boundaries, and metrics from mask.

    Returns (centers, lefts, rights, metrics) where each point is (x, y)
    in full-image coordinates.
    """
    img_h, img_w = mask.shape
    roi_top = int(img_h * roi_top_pct)
    roi = mask[roi_top:, :]
    roi_h, roi_w = roi.shape

    centers = []
    lefts = []
    rights = []
    widths = []

    for r in range(0, roi_h, sample_step):
        row = roi[r, :]
        trav = np.where(row > 127)[0]
        if len(trav) == 0:
            continue
        left = int(trav[0])
        right = int(trav[-1])
        center = (left + right) // 2
        width = right - left
        y = roi_top + r
        centers.append((center, y))
        lefts.append((left, y))
        rights.append((right, y))
        widths.append(width)

    # Smooth centerline
    if len(centers) >= smooth_window:
        xs = np.array([c[0] for c in centers], dtype=np.float64)
        kernel = np.ones(smooth_window) / smooth_window
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


def draw_path(display, centers, lefts, rights):
    """Draw planned path centerline and corridor boundaries on the display."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found: {ckpt_path.resolve()}")
        sys.exit(1)

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

    try:
        map_name = args.map
        print(f"[INFO] Loading map: {map_name}")
        scenario = Scenario(map_name, "offroad_path_overlay")

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

        # ── State ─────────────────────────────────────────────────────────────
        show_overlay = True
        show_path = True
        ai_driving = False
        recording = False
        video_writer = None
        record_start = None
        output_dir = Path(args.output_dir)

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        # Manual control state
        throttle = 0.0
        brake = 0.0
        steering = 0.0

        cv2.namedWindow("BeamNG Path Overlay", cv2.WINDOW_NORMAL)

        print("\n" + "=" * 55)
        print("  BeamNG Off-Road Path Overlay")
        print("=" * 55)
        print("  W/S = Throttle/Brake   A/D = Steer")
        print("  SPACE = Handbrake      P = AI Drive")
        print("  T = Traversability     O = Path Overlay")
        print("  R = Record             G = Get Position")
        print("  Q/ESC = Quit")
        print("=" * 55 + "\n")

        # ── Main loop ─────────────────────────────────────────────────────────
        while True:
            images = camera.stream()
            colour = images.get("colour")
            if colour is None:
                time.sleep(0.01)
                continue

            frame_rgb = np.array(colour)
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Always run inference when either overlay is active
            mask = None
            if show_overlay or show_path:
                x = preprocess(bgr, args.img_size, device)
                mask_small = predict_mask(model, x, args.threshold)
                mask = cv2.resize(mask_small, (bgr.shape[1], bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            # Build display
            if show_overlay and mask is not None:
                display = overlay_green(bgr, mask, args.alpha)
            else:
                display = bgr.copy()

            # Extract and draw path
            if show_path and mask is not None:
                centers, lefts, rights, metrics = extract_path(mask)
                draw_path(display, centers, lefts, rights)
            else:
                metrics = {"traversable_pct": 0, "avg_corridor_width": 0,
                           "min_corridor_width": 0}

            # Recording
            if recording and video_writer:
                side_by_side = np.hstack([bgr, display])
                video_writer.write(side_by_side)

            # FPS counter
            fps_counter += 1
            fps_elapsed = time.time() - fps_timer
            if fps_elapsed >= 1.0:
                current_fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_timer = time.time()

            # HUD
            h, w = display.shape[:2]
            rec_text = "REC" if recording else ""
            rec_time = f" {time.time() - record_start:.1f}s" if recording and record_start else ""
            overlay_text = "ON" if show_overlay else "OFF"
            path_text = "ON" if show_path else "OFF"
            ai_text = "AI" if ai_driving else "Manual"
            trav_pct = metrics["traversable_pct"] * 100
            corr_w = metrics["avg_corridor_width"]

            hud = (f"FPS: {current_fps:.0f}  Overlay: {overlay_text}  "
                   f"Path: {path_text}  {ai_text}  {rec_text}{rec_time}")
            cv2.putText(display, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
            cv2.putText(display, hud, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            line2 = f"Traversable: {trav_pct:.0f}%  Corridor: {corr_w:.0f}px"
            cv2.putText(display, line2, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(display, line2, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 2)

            if recording:
                cv2.circle(display, (w - 25, 25), 10, (0, 0, 255), -1)

            controls = ("W/S=Throttle  A/D=Steer  P=AI  T=Overlay  "
                        "O=Path  R=Rec  G=GetPos  Q=Quit")
            cv2.putText(display, controls, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("BeamNG Path Overlay", display)
            key = cv2.waitKey(1) & 0xFF

            # ── Keyboard ──────────────────────────────────────────────────────
            if key == ord("q") or key == 27:
                break

            elif key == ord("t"):
                show_overlay = not show_overlay
                print(f"[INFO] Traversability overlay: {'ON' if show_overlay else 'OFF'}")

            elif key == ord("o"):
                show_path = not show_path
                print(f"[INFO] Path overlay: {'ON' if show_path else 'OFF'}")

            elif key == ord("p"):
                ai_driving = not ai_driving
                if ai_driving:
                    vehicle.ai.set_mode("random")
                    vehicle.ai.set_speed(args.ai_speed, mode="limit")
                    print(f"[INFO] AI driving: ON (random, {args.ai_speed} m/s)")
                else:
                    vehicle.ai.set_mode("disabled")
                    throttle = 0.0
                    brake = 0.0
                    steering = 0.0
                    print("[INFO] AI driving: OFF (manual)")

            elif key == ord("g"):
                vehicle.sensors.poll()
                pos = vehicle.state.get("pos", "unknown")
                rot = vehicle.state.get("rotation", "unknown")
                print(f"[POS] Position: {pos}  Rotation: {rot}")

            elif key == ord("r"):
                if not recording:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_path = output_dir / f"beamng_pathoverlay_{ts}.mp4"
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

            # Manual driving (only when AI is off)
            elif not ai_driving:
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
