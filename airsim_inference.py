#!/usr/bin/env python3
"""
airsim_inference.py

AirSim semantic-segmentation inference (NO screen-capture).
- Pulls frames directly from AirSim via simGetImages()
- Loads a segmentation model from a PyTorch checkpoint
- Runs inference and shows an overlay window
- Press R to record side-by-side video (raw | segmented)

Example:
  python airsim_inference.py \
    --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth \
    --arch deeplabv3p \
    --encoder resnet50 \
    --camera 0 \
    --viz

Controls (in the OpenCV window):
    R       - Toggle recording (side-by-side raw + segmented video)
    Q/ESC   - Quit
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import airsim

try:
    import cv2
except Exception as e:
    raise RuntimeError("opencv-python is required: pip install opencv-python") from e

try:
    import segmentation_models_pytorch as smp
except Exception as e:
    raise RuntimeError(
        "segmentation-models-pytorch is required for --arch/--encoder loading.\n"
        "Install: pip install segmentation-models-pytorch"
    ) from e


ARCH_ALIASES = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "fpn": "FPN",
    "pspnet": "PSPNet",
    "pan": "PAN",
    "deeplabv3": "DeepLabV3",
    "deeplabv3p": "DeepLabV3Plus",
    "deeplabv3plus": "DeepLabV3Plus",
}


def parse_args():
    p = argparse.ArgumentParser()

    # AirSim
    p.add_argument("--ip", default="127.0.0.1")
    p.add_argument("--vehicle", default="Car1")
    p.add_argument("--camera", default="0")
    p.add_argument("--image-type", default="scene", choices=["scene"])
    p.add_argument("--fps", type=float, default=30.0)

    # Model
    p.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    p.add_argument("--arch", required=True, help="e.g. deeplabv3p, unet, fpn")
    p.add_argument("--encoder", required=True, help="e.g. resnet50, mit_b0, etc.")
    p.add_argument("--encoder-weights", default="imagenet", help="e.g. imagenet or None")
    p.add_argument("--classes", type=int, default=1, help="1 for binary; >1 for multiclass")
    p.add_argument("--activation", default=None, help="Leave None; we apply sigmoid/softmax ourselves")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for binary mask")
    p.add_argument("--drivable-class", type=int, default=1, help="For multiclass: which class index is drivable")

    # Viz / save
    p.add_argument("--viz", action="store_true", help="Show overlay window")
    p.add_argument("--alpha", type=float, default=0.45, help="Overlay alpha")
    p.add_argument("--output-dir", default="airsim_recordings", help="Directory for recorded videos")

    return p.parse_args()


def get_device(flag: str) -> torch.device:
    if flag == "cpu":
        return torch.device("cpu")
    if flag == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_smp_model(arch: str, encoder: str, encoder_weights, classes: int, activation):
    arch_key = arch.strip().lower()
    if arch_key not in ARCH_ALIASES:
        raise ValueError(
            f"Unknown --arch '{arch}'. Supported: {', '.join(sorted(ARCH_ALIASES.keys()))}"
        )
    cls_name = ARCH_ALIASES[arch_key]
    ModelCls = getattr(smp, cls_name)

    if encoder_weights in ["None", "none", "null", ""]:
        encoder_weights = None

    model = ModelCls(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=classes,
        activation=activation,
    )
    return model


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    state_dict = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None:
            if all(hasattr(v, "shape") for v in ckpt.values()):
                state_dict = ckpt
    else:
        state_dict = ckpt

    if state_dict is None or not isinstance(state_dict, dict):
        raise RuntimeError(f"Could not find a state_dict inside checkpoint: {ckpt_path}")

    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys ({len(missing)}). First 10: {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys ({len(unexpected)}). First 10: {unexpected[:10]}")


def airsim_request(camera: str) -> airsim.ImageRequest:
    return airsim.ImageRequest(camera, airsim.ImageType.Scene, pixels_as_float=False, compress=False)


def airsim_to_bgr(resp: airsim.ImageResponse) -> np.ndarray:
    if resp.width == 0 or resp.height == 0:
        raise RuntimeError("Empty image from AirSim (width/height=0). Check camera name/settings.json.")
    n_bytes = len(resp.image_data_uint8)
    n_pixels = resp.height * resp.width
    channels = n_bytes // n_pixels
    img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(resp.height, resp.width, channels)
    bgr = img[:, :, :3]
    return bgr


def preprocess_bgr_to_tensor(bgr: np.ndarray, input_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    h, w = input_hw
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(rgb).to(device=device, dtype=torch.float32) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3)
    x = (x - mean) / std
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    x: torch.Tensor,
    classes: int,
    threshold: float,
    drivable_class: int,
) -> np.ndarray:
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        y = y.get("out", next(iter(y.values())))

    if y.dim() != 4:
        raise RuntimeError(f"Unexpected model output shape: {tuple(y.shape)}")

    if classes == 1:
        prob = torch.sigmoid(y)[0, 0]
        mask = (prob >= threshold).to(torch.uint8) * 255
    else:
        cls = torch.argmax(y, dim=1)[0].to(torch.int64)
        mask = (cls == int(drivable_class)).to(torch.uint8) * 255

    return mask.cpu().numpy()


def overlay_green(bgr: np.ndarray, mask_255: np.ndarray, alpha: float) -> np.ndarray:
    if mask_255.shape[:2] != bgr.shape[:2]:
        mask_255 = cv2.resize(mask_255, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    out = bgr.copy()
    green = np.zeros_like(out)
    green[:, :, 1] = 255
    m = (mask_255 > 0)[:, :, None]

    out = np.where(m, (alpha * green + (1 - alpha) * out).astype(np.uint8), out)
    return out


def main():
    args = parse_args()
    device = get_device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    # Build + load model
    model = build_smp_model(
        arch=args.arch,
        encoder=args.encoder,
        encoder_weights=args.encoder_weights,
        classes=args.classes,
        activation=args.activation,
    )
    model.to(device)
    load_checkpoint_into_model(model, ckpt_path, device)
    model.eval()

    # AirSim connect
    client = airsim.CarClient(ip=args.ip)
    client.confirmConnection()
    # Don't enable API control — let the user drive manually in AirSim window

    req = airsim_request(args.camera)

    frame_idx = 0
    recording = False
    video_writer = None
    record_start = None
    output_dir = Path(args.output_dir)

    # FPS tracking
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0

    if args.viz:
        cv2.namedWindow("AirSim Segmentation", cv2.WINDOW_NORMAL)

    print(f"[INFO] AirSim connected: vehicle={args.vehicle}, camera={args.camera}")
    print(f"[INFO] Model: arch={args.arch}, encoder={args.encoder}, classes={args.classes}, device={device}")
    print(f"[INFO] FPS target: {args.fps}")
    print("[INFO] Controls: R=Record, Q=Quit")

    try:
        while True:
            t0 = time.time()

            # Get image from AirSim
            responses = client.simGetImages([req])
            bgr = airsim_to_bgr(responses[0])

            # Inference
            x = preprocess_bgr_to_tensor(bgr, tuple(args.input_size), device)
            mask_small = predict_mask(model, x, args.classes, args.threshold, args.drivable_class)

            # Resize mask to frame
            mask = cv2.resize(mask_small, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Overlay
            vis = overlay_green(bgr, mask, args.alpha)

            # Build side-by-side frame for recording: [raw | segmented]
            if recording and video_writer:
                side_by_side = np.hstack([bgr, vis])
                video_writer.write(side_by_side)

            # FPS calculation
            fps_counter += 1
            fps_elapsed = time.time() - fps_timer
            if fps_elapsed >= 1.0:
                current_fps = fps_counter / fps_elapsed
                fps_counter = 0
                fps_timer = time.time()

            # Add HUD to display
            display = vis.copy()
            h, w = display.shape[:2]
            rec_text = "REC" if recording else ""
            rec_time = f" {time.time() - record_start:.1f}s" if recording and record_start else ""
            hud = f"FPS: {current_fps:.0f}  {rec_text}{rec_time}"
            cv2.putText(display, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
            cv2.putText(display, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if recording:
                cv2.circle(display, (w - 25, 25), 10, (0, 0, 255), -1)
            cv2.putText(display, "R=Record  Q=Quit", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            # Show
            if args.viz:
                cv2.imshow("AirSim Segmentation", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("r"):
                    if not recording:
                        # Start recording
                        output_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        video_path = output_dir / f"airsim_recording_{ts}.mp4"
                        frame_h, frame_w = bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(video_path), fourcc, args.fps, (frame_w * 2, frame_h)
                        )
                        recording = True
                        record_start = time.time()
                        print(f"[REC] Recording started: {video_path}")
                    else:
                        # Stop recording
                        recording = False
                        if video_writer:
                            video_writer.release()
                            video_writer = None
                        elapsed = time.time() - record_start if record_start else 0
                        print(f"[REC] Recording stopped ({elapsed:.1f}s)")
                        record_start = None

            frame_idx += 1

            # Cap to target FPS so recordings play at correct speed
            elapsed = time.time() - t0
            sleep_time = (1.0 / args.fps) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        if video_writer:
            video_writer.release()
            print("[REC] Recording saved.")
        if args.viz:
            cv2.destroyAllWindows()
        print("[INFO] Stopped cleanly.")


if __name__ == "__main__":
    main()
