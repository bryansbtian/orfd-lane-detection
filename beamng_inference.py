"""
BeamNG.tech Inference Script for Off-Road Lane Detection

Launches BeamNG.tech, loads an off-road map, attaches a front camera,
and runs real-time semantic segmentation inference.

Usage:
    python beamng_inference.py \
        --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth \
        --arch deeplabv3p --encoder resnet50

Controls (in the OpenCV window):
    W/S     - Throttle/Brake
    A/D     - Steer left/right
    SPACE   - Handbrake
    P       - Toggle AI driving (random waypoints)
    R       - Toggle recording
    T       - Toggle lane detection overlay
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


# ── Model helpers ──────────────────────────────────────────────────────────────

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

# BeamNG off-road maps and good spawn positions (x, y, z)
OFFROAD_MAPS = {
    "jungle_rock_island": {
        "spawns": [
            ((122.03, -74.66, 158.65), (0, 0, 0.966, 0.259)),  # Paved jungle road
            ((353.80, -151.3, 160.19), (0, 0, -0.875, 0.484)),  # Jungle hill road
        ],
    },
    "johnson_valley": {
        "spawns": [
            ((-1152, 291.05, 119.84), (0, 0, 0, 1)),      # Desert dirt road
        ],
    },
    "west_coast_usa": {
        "spawns": [
            ((565.13, 27.664, 148.06), (0, 0, -0.934, 0.355)),  # Highway overpass
        ],
    },
    "utah": {
        "spawns": [
            ((575, 118, 132), (0, 0, 0, 1)),              # Desert dirt road
            ((200, 200, 135), (0, 0, 0.707, 0.707)),      # Off-road trail
        ],
    },
    "small_island": {
        "spawns": [
            ((240, 215, 37), (0, 0, 0, 1)),               # Island dirt path
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
    p = argparse.ArgumentParser(description="BeamNG.tech off-road lane detection inference")

    # BeamNG
    p.add_argument("--beamng-home", type=str,
                    default=r"E:\BeamNG.tech.v0.38.3.0",
                    help="Path to BeamNG.tech installation")
    p.add_argument("--beamng-user", type=str, default=None,
                    help="Path for BeamNG user folder (default: auto)")
    p.add_argument("--map", type=str, default="jungle_rock_island",
                    help="Map to load (e.g. johnson_valley, jungle_rock_island, utah)")
    p.add_argument("--spawn-idx", type=int, default=0,
                    help="Spawn point index (from map's built-in spawn points)")
    p.add_argument("--vehicle", type=str, default="pickup",
                    help="Vehicle model (pickup, roamer, etc.)")

    # Camera
    p.add_argument("--width", type=int, default=1280, help="Camera width")
    p.add_argument("--height", type=int, default=720, help="Camera height")
    p.add_argument("--fov", type=float, default=70, help="Camera vertical FOV")
    p.add_argument("--fps", type=float, default=30.0, help="Target FPS")

    # Model
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to .pth model checkpoint")
    p.add_argument("--arch", type=str, default="deeplabv3p",
                    help="Model architecture (unet, deeplabv3p, fpn, etc.)")
    p.add_argument("--encoder", type=str, default="resnet50",
                    help="Encoder backbone (resnet34, resnet50, etc.)")
    p.add_argument("--img-size", type=int, default=256,
                    help="Model input size (lower = faster, 256 recommended for real-time)")
    p.add_argument("--threshold", type=float, default=0.5,
                    help="Sigmoid threshold for binary mask")

    # Output
    p.add_argument("--output-dir", type=str, default="beamng_recordings",
                    help="Directory for recorded videos")
    p.add_argument("--alpha", type=float, default=0.45,
                    help="Overlay transparency")

    return p.parse_args()


def build_model(arch, encoder, encoder_weights="imagenet"):
    arch_key = arch.strip().lower()
    if arch_key not in ARCH_ALIASES:
        raise ValueError(f"Unknown arch '{arch}'. Supported: {sorted(ARCH_ALIASES.keys())}")
    cls_name = ARCH_ALIASES[arch_key]
    ModelCls = getattr(smp, cls_name)
    model = ModelCls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=1,
        activation=None,
    )
    return model


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
        model.half()  # FP16 for faster inference
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
        # Load the map and spawn vehicle at a safe temporary position
        map_name = args.map
        print(f"[INFO] Loading map: {map_name}")
        scenario = Scenario(map_name, "offroad_inference")

        # Use predefined spawn if available, otherwise fall back to waypoints
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
        # In BeamNG, vehicle forward is -Y, so dir=(0, -1, 0)
        camera = Camera(
            name="front_cam",
            bng=bng,
            vehicle=vehicle,
            pos=(0, -2.5, 0.8),       # Ahead of front bumper, slightly above ground
            dir=(0, -1, -0.1),        # Looking forward and slightly down
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
        print(f"[INFO] Vehicle: {args.vehicle}")

        # ── Main loop ─────────────────────────────────────────────────────────
        show_overlay = True
        recording = False
        video_writer = None
        record_start = None
        output_dir = Path(args.output_dir)
        ai_driving = False

        fps_counter = 0
        fps_timer = time.time()
        current_fps = 0.0

        # Control state
        throttle = 0.0
        brake = 0.0
        steering = 0.0

        cv2.namedWindow("BeamNG Off-Road Detection", cv2.WINDOW_NORMAL)

        print("\n" + "=" * 55)
        print("  BeamNG Off-Road Lane Detection")
        print("=" * 55)
        print("  W/S = Throttle/Brake   A/D = Steer")
        print("  SPACE = Handbrake      P = AI Drive")
        print("  R = Record             T = Toggle Overlay")
        print("  Q/ESC = Quit")
        print("=" * 55 + "\n")

        while True:
            t0 = time.time()

            # Stream camera (no roundtrip, reads shared memory directly)
            images = camera.stream()
            colour = images.get("colour")
            if colour is None:
                time.sleep(0.01)
                continue

            # Convert PIL image to BGR numpy
            frame_rgb = np.array(colour)
            bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Run inference
            if show_overlay:
                x = preprocess(bgr, args.img_size, device)
                mask_small = predict_mask(model, x, args.threshold)
                mask = cv2.resize(mask_small, (bgr.shape[1], bgr.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
                display = overlay_green(bgr, mask, args.alpha)
            else:
                display = bgr.copy()

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
            ai_text = "AI" if ai_driving else "Manual"
            hud = f"FPS: {current_fps:.0f}  Overlay: {overlay_text}  {ai_text}  {rec_text}{rec_time}"

            cv2.putText(display, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if recording:
                cv2.circle(display, (w - 25, 25), 10, (0, 0, 255), -1)

            controls_text = "W/S=Throttle  A/D=Steer  P=AI  R=Rec  T=Overlay  G=GetPos  Q=Quit"
            cv2.putText(display, controls_text, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("BeamNG Off-Road Detection", display)
            key = cv2.waitKey(1) & 0xFF

            # ── Keyboard handling ─────────────────────────────────────────────
            if key == ord("q") or key == 27:
                break

            elif key == ord("t"):
                show_overlay = not show_overlay
                print(f"[INFO] Overlay: {'ON' if show_overlay else 'OFF'}")

            elif key == ord("p"):
                ai_driving = not ai_driving
                if ai_driving:
                    vehicle.ai.set_mode("random")
                    vehicle.ai.set_speed(8, mode="limit")
                    print("[INFO] AI driving: ON (random waypoints)")
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
                    video_path = output_dir / f"beamng_recording_{ts}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, args.fps,
                        (args.width * 2, args.height)
                    )
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

            # Manual driving controls (only when AI is off)
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

                # Decay steering toward center
                steering *= 0.92
                if abs(steering) < 0.01:
                    steering = 0.0

                vehicle.control(steering=steering, throttle=throttle, brake=brake)

            # Minimal wait for OpenCV event processing
            pass

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
