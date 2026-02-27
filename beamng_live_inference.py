#!/usr/bin/env python3
"""
BeamNG Live Inference — YOLO26 vs SAM3 Side-by-Side Comparison
===============================================================
Runs real-time road-segmentation on a BeamNG.tech camera feed and shows
YOLO26 (left panel) and SAM3 (right panel) side-by-side for direct comparison.

Each model runs in its own background thread so the display stays live at
camera speed even when SAM3 is slow (~300 ms/frame).

Controls
--------
  W / S     — Throttle / Brake
  A / D     — Steer left / right
  SPACE     — Handbrake
  P         — Toggle AI driving (random waypoints)
  T         — Toggle segmentation overlay (mask on/off)
  R         — Start / stop recording  →  output/beamng_comparison_<ts>.mp4
  S         — Save screenshot          →  output/beamng_screenshot_<ts>.png
  G         — Print current vehicle position (useful for finding spawn points)
  Q / ESC   — Quit

Usage
-----
  python beamng_live_inference.py                          # both models (default)
  python beamng_live_inference.py --model yolo26           # YOLO26 only
  python beamng_live_inference.py --model sam3             # SAM3 only
  python beamng_live_inference.py --map utah --spawn-idx 1
  python beamng_live_inference.py --capture screen         # screen-capture fallback
"""

import argparse
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Project helpers ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from segment_road import (
    YOLO26Segmentor,
    SAM3Segmentor,
    extract_masks,
    overlay_mask,
    draw_hud,
    compute_metrics,
)

# ── Off-road maps with tested spawn points ─────────────────────────────────────
OFFROAD_MAPS = {
    "jungle_rock_island": {
        "spawns": [
            ((122.03, -74.66,  158.65), (0, 0,  0.966,  0.259)),
            ((353.80, -151.3,  160.19), (0, 0, -0.875,  0.484)),
        ],
    },
    "johnson_valley": {
        "spawns": [
            ((-1152,  291.05, 119.84), (0, 0, 0, 1)),
        ],
    },
    "west_coast_usa": {
        "spawns": [
            ((565.13, 27.664, 148.06), (0, 0, -0.934, 0.355)),
        ],
    },
    "utah": {
        "spawns": [
            ((575,   118,    132),     (0, 0, 0,     1)),
            ((200,   200,    135),     (0, 0, 0.707, 0.707)),
        ],
    },
    "small_island": {
        "spawns": [
            ((240,   215,     37),     (0, 0, 0,     1)),
            ((145,   240,     37),     (0, 0, 0.707, 0.707)),
        ],
    },
    "east_coast_usa": {
        "spawns": [
            ((-700,   50,     35),     (0, 0, 0, 1)),
        ],
    },
}

# ── Constants ──────────────────────────────────────────────────────────────────
BEAMNG_HOME  = r"E:\BeamNG.tech.v0.38.3.0"
BEAMNG_HOST  = "localhost"
BEAMNG_PORT  = 64256

DEFAULT_MAP  = "jungle_rock_island"
DEFAULT_VEH  = "pickup"
DEFAULT_W    = 1280
DEFAULT_H    = 720
DEFAULT_FOV  = 120.0

OUTPUT_DIR   = Path("output")
DIVIDER_W    = 6
DIVIDER_BGR  = (0, 230, 230)
STATUS_H     = 34
RECORD_FPS   = 15


# ── Background inference worker ────────────────────────────────────────────────

class InferenceWorker(threading.Thread):
    """
    Runs one segmentation model continuously in a background thread.

    Frames are downscaled to (infer_w × infer_h) before inference; the output
    mask is upscaled back to full display resolution so the overlay stays sharp.
    A two-pass warmup eliminates the JIT / CUDA-allocation spike on the first
    real frame.

    The main thread calls submit_frame() (non-blocking) and get_result()
    (non-blocking) so it is never stalled by slow models like SAM3.
    """

    def __init__(self, seg, model_name: str, infer_w: int, infer_h: int):
        super().__init__(daemon=True, name=f"worker-{model_name}")
        self.seg        = seg
        self.model_name = model_name
        self.infer_w    = infer_w
        self.infer_h    = infer_h

        self._in_q      = queue.Queue(maxsize=1)
        self._lock      = threading.Lock()
        self._panel     = None
        self._mask      = None
        self._ready     = False
        self._stop      = threading.Event()
        self.start_time = None   # set when worker enters main loop (after warmup)

    def submit_frame(self, frame: np.ndarray, prev_mask, show_overlay: bool):
        """Non-blocking submit; drops the waiting frame so workers stay current."""
        item = (frame, prev_mask, show_overlay)
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._in_q.put_nowait(item)
        except queue.Full:
            pass

    def get_result(self):
        """Return (panel_bgr, mask) — latest completed result, non-blocking."""
        with self._lock:
            return self._panel, self._mask

    @property
    def ready(self) -> bool:
        return self._ready

    def stop(self):
        self._stop.set()
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass

    def run(self):
        # ── Warmup ────────────────────────────────────────────────────────────
        # Use random-noise frames so all CUDA kernels and memory paths are
        # triggered during warmup rather than on the first real camera frame.
        # An all-black (zeros) dummy skips many branches → 30 s JIT stall on
        # the first live frame.  5 passes with varied random noise cover them.
        print(f"[{self.model_name}] Warming up ({self.infer_w}x{self.infer_h}, 5 passes) ...")
        rng = np.random.default_rng(0)
        for i in range(5):
            dummy = rng.integers(0, 256, (self.infer_h, self.infer_w, 3),
                                 dtype=np.uint8)
            try:
                self.seg.infer(dummy)
                print(f"[{self.model_name}]   pass {i+1}/5 done")
            except Exception as exc:
                print(f"[{self.model_name}]   pass {i+1}/5 warning: {exc}")

        # Flush any remaining CUDA work so the main loop starts clean
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        print(f"[{self.model_name}] Worker ready")
        self.start_time = time.perf_counter()

        # ── Main inference loop ───────────────────────────────────────────────
        while not self._stop.is_set():
            try:
                item = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            frame, prev_mask, show_overlay = item
            disp_h, disp_w = frame.shape[:2]

            try:
                # Downscale → inference → upscale mask to display resolution
                if self.infer_w != disp_w or self.infer_h != disp_h:
                    infer_frame = cv2.resize(
                        frame, (self.infer_w, self.infer_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    infer_frame = frame

                result, t_ms = self.seg.infer(infer_frame)
                # Pass display dims → extract_masks upscales mask to full res
                mask, confs  = extract_masks(result, disp_h, disp_w)
                m     = compute_metrics(mask, confs, t_ms, prev_mask)
                panel = overlay_mask(frame, mask) if show_overlay else frame.copy()
                panel = draw_hud(panel, m, self.model_name.upper())

            except Exception as exc:
                print(f"[{self.model_name}] Inference error: {exc}")
                panel = frame.copy()
                mask  = np.zeros((disp_h, disp_w), dtype=bool)

            with self._lock:
                first = not self._ready
                self._panel = panel
                self._mask  = mask
                self._ready = True
            if first:
                elapsed = time.perf_counter() - self.start_time
                print(f"[{self.model_name}] First result ready ({elapsed:.1f}s after warmup)")


# ── BeamNG setup ───────────────────────────────────────────────────────────────

def connect_beamng(args):
    """Launch BeamNG, load map, spawn vehicle, attach streaming camera."""
    from beamngpy import BeamNGpy, Scenario, Vehicle
    from beamngpy.sensors.camera import Camera

    print(f"[BeamNG] Launching from {args.beamng_home} …")
    bng = BeamNGpy(BEAMNG_HOST, BEAMNG_PORT, home=args.beamng_home)
    bng.open(launch=True)

    map_name = args.map
    map_cfg  = OFFROAD_MAPS.get(map_name)
    if map_cfg:
        idx = min(args.spawn_idx, len(map_cfg["spawns"]) - 1)
        spawn_pos, spawn_rot = map_cfg["spawns"][idx]
        print(f"[BeamNG] Spawn #{idx}: {spawn_pos}")
    else:
        spawn_pos, spawn_rot = (0, 0, 0), (0, 0, 0, 1)
        print(f"[BeamNG] Unknown map '{map_name}' — using origin spawn")

    print(f"[BeamNG] Loading map '{map_name}' …")
    scenario = Scenario(map_name, "orfd_live_inference")
    vehicle  = Vehicle("ego", model=args.vehicle, licence="OFFROAD")
    scenario.add_vehicle(vehicle, pos=spawn_pos, rot_quat=spawn_rot, cling=True)

    scenario.make(bng)
    bng.scenario.load(scenario)
    bng.scenario.start()
    vehicle.connect(bng)

    print("[BeamNG] Waiting for physics to settle …")
    time.sleep(3.0)

    print("[BeamNG] Attaching camera …")
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
    time.sleep(1.0)
    print("[BeamNG] Camera ready.\n")
    return bng, vehicle, camera


def get_beamng_frame(camera) -> np.ndarray | None:
    try:
        images = camera.stream()
        colour = images.get("colour")
        if colour is None:
            return None
        return cv2.cvtColor(np.array(colour.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as exc:
        print(f"[Cam] {exc}")
        return None


# ── Screen-capture fallback ────────────────────────────────────────────────────

def get_screen_frame(mss_lib, width, height) -> np.ndarray | None:
    try:
        with mss_lib.mss() as sct:
            shot = np.array(sct.grab(sct.monitors[1]))
        return cv2.resize(cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR), (width, height))
    except Exception as exc:
        print(f"[Screen] {exc}")
        return None


# ── Display helpers ────────────────────────────────────────────────────────────

def make_canvas(panels: list) -> np.ndarray:
    if len(panels) == 1:
        return panels[0]
    div = np.full((panels[0].shape[0], DIVIDER_W, 3), DIVIDER_BGR, dtype=np.uint8)
    return np.hstack([panels[0], div, panels[1]])


def draw_status_bar(canvas, recording, rec_elapsed, ai_driving, show_overlay, fps):
    w   = canvas.shape[1]
    bar = np.zeros((STATUS_H, w, 3), dtype=np.uint8)
    x   = 8

    def put(txt, col, bold=False):
        nonlocal x
        th = 2 if bold else 1
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, th)
        cv2.putText(bar, txt, (x, STATUS_H - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, th, cv2.LINE_AA)
        x += tw + 16

    if recording:
        put(f"[REC {rec_elapsed:.0f}s]", (0, 60, 255), bold=True)
    else:
        put("[ R ] Rec",     (150, 150, 150))

    put("[ S ] Shot",        (150, 150, 150))
    put(f"[ T ] Overlay {'ON' if show_overlay else 'OFF'}",
        (100, 255, 100) if show_overlay else (150, 150, 150))
    put(f"[ P ] {'AI' if ai_driving else 'Manual'}",
        (100, 200, 255) if ai_driving else (150, 150, 150))
    put("WASD/SPC",          (150, 150, 150))
    put("[ G ] Pos",         (150, 150, 150))
    put("[ Q ] Quit",        (150, 150, 150))

    fps_txt = f"FPS: {fps:.1f}"
    (fw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(bar, fps_txt, (w - fw - 10, STATUS_H - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1, cv2.LINE_AA)

    return np.vstack([canvas, bar])


# ── Waiting splash while workers warm up ──────────────────────────────────────

def make_waiting_panel(frame: np.ndarray, model_name: str, elapsed: float) -> np.ndarray:
    """Show the raw frame with a waiting label until first result is ready."""
    panel = frame.copy()
    cv2.putText(panel, f"{model_name.upper()}: waiting for first result... ({elapsed:.0f}s)",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
    return panel


# ── Main ───────────────────────────────────────────────────────────────────────

def run(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────────
    models = {}
    if args.model in ("yolo26", "both"):
        print("[Init] Loading YOLO26 …")
        models["yolo26"] = YOLO26Segmentor(
            size=args.yolo_size, conf=args.conf, prompts=args.prompts or None,
        )
    if args.model in ("sam3", "both"):
        print("[Init] Loading SAM3 …")
        models["sam3"] = SAM3Segmentor(
            weights=args.sam3_weights, conf=args.conf, prompts=args.prompts or None,
        )
    if not models:
        sys.exit("[ERROR] No models loaded.")

    model_names = list(models.keys())

    # ── Compute inference resolution (width × scaled height, keeps aspect) ───
    infer_w = args.infer_size
    infer_h = max(1, infer_w * args.height // args.width)
    print(f"[Init] Inference resolution: {infer_w}×{infer_h}  "
          f"(display: {args.width}×{args.height})")

    # ── Start inference worker threads ─────────────────────────────────────────
    workers = {
        name: InferenceWorker(seg, name, infer_w, infer_h)
        for name, seg in models.items()
    }
    for w in workers.values():
        w.start()
        print(f"[Thread] {w.name} started")

    # ── Connect to BeamNG / screen ─────────────────────────────────────────────
    bng = vehicle = camera = mss_lib = None

    if args.capture == "beamngpy":
        bng, vehicle, camera = connect_beamng(args)
    else:
        try:
            import mss as _mss
            mss_lib = _mss
        except ImportError:
            sys.exit("\n[ERROR] pip install mss\n")
        print("[Screen] Make sure BeamNG.tech window is open and maximised.")

    # ── OpenCV window ──────────────────────────────────────────────────────────
    win = "BeamNG Live Inference — YOLO26 vs SAM3"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    canvas_w = args.width * len(models) + DIVIDER_W * max(0, len(models) - 1)
    cv2.resizeWindow(win, min(canvas_w, 1800), min(args.height + STATUS_H, 980))

    print("\n" + "=" * 58)
    print("  BeamNG Live Inference — YOLO26 vs SAM3")
    print("=" * 58)
    print("  W/S=Throttle/Brake   A/D=Steer   SPC=Brake")
    print("  P=AI toggle          T=Overlay   G=Print pos")
    print("  R=Record             S=Screenshot   Q=Quit")
    print("=" * 58 + "\n")

    # ── State ──────────────────────────────────────────────────────────────────
    show_overlay = True
    recording    = False
    writer       = None
    out_path     = None
    rec_start    = None
    ai_driving   = False
    throttle     = 0.0
    brake        = 0.0
    steering     = 0.0

    prev_masks   = {n: None for n in model_names}
    frame_count  = 0
    fps_times    = deque(maxlen=30)
    last_frame   = None   # cached raw frame for display when no new frame arrives

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF   # ← must be called every iteration

            # ── Quit ───────────────────────────────────────────────────────────
            if key in (ord("q"), 27):
                break

            # ── Overlay toggle ─────────────────────────────────────────────────
            if key == ord("t"):
                show_overlay = not show_overlay
                print(f"[Overlay] {'ON' if show_overlay else 'OFF'}")

            # ── AI driving ─────────────────────────────────────────────────────
            if key == ord("p") and vehicle is not None:
                ai_driving = not ai_driving
                if ai_driving:
                    vehicle.ai.set_mode("random")
                    vehicle.ai.set_speed(8, mode="limit")
                    print("[AI] ON")
                else:
                    vehicle.ai.set_mode("disabled")
                    throttle = brake = steering = 0.0
                    vehicle.control(steering=0, throttle=0, brake=0)
                    print("[AI] OFF")

            # ── Print position ─────────────────────────────────────────────────
            if key == ord("g") and vehicle is not None:
                try:
                    vehicle.sensors.poll()
                    print(f"[POS] {vehicle.state.get('pos')}  rot={vehicle.state.get('rotation')}")
                except Exception as exc:
                    print(f"[POS] {exc}")

            # ── Recording toggle ───────────────────────────────────────────────
            if key == ord("r"):
                if not recording:
                    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                    out_path = OUTPUT_DIR / f"beamng_comparison_{ts}.mp4"
                    recording = True
                    writer    = None
                    rec_start = time.time()
                    print(f"[Rec] Started → {out_path.name}")
                else:
                    recording = False
                    if writer is not None:
                        writer.release()
                        writer = None
                    print(f"[Rec] Stopped ({time.time()-rec_start:.1f}s) → {out_path.name}")
                    rec_start = None

            take_shot = (key == ord("s"))

            # ── Manual vehicle controls ────────────────────────────────────────
            if vehicle is not None and not ai_driving:
                if key == ord("w"):
                    throttle = min(throttle + 0.15, 1.0); brake = 0.0
                elif key == ord("s"):
                    brake = min(brake + 0.15, 1.0);       throttle = 0.0
                elif key == ord("a"):
                    steering = max(steering - 0.10, -1.0)
                elif key == ord("d"):
                    steering = min(steering + 0.10,  1.0)
                elif key == ord(" "):
                    brake = 1.0; throttle = 0.0

                steering *= 0.92
                if abs(steering) < 0.01:
                    steering = 0.0
                vehicle.control(steering=steering, throttle=throttle, brake=brake)

            # ── Capture new frame ──────────────────────────────────────────────
            if args.capture == "beamngpy":
                frame = get_beamng_frame(camera)
            else:
                frame = get_screen_frame(mss_lib, args.width, args.height)

            if frame is not None:
                last_frame = frame
                # Submit to each worker (non-blocking)
                for name, worker in workers.items():
                    worker.submit_frame(frame, prev_masks[name], show_overlay)
            elif last_frame is None:
                # No frame yet at all — spin
                time.sleep(0.02)
                continue

            display_frame = last_frame

            # ── Collect latest results from workers (non-blocking) ─────────────
            panels = []
            for name, worker in workers.items():
                panel, mask = worker.get_result()
                if mask is not None:
                    prev_masks[name] = mask
                if panel is not None:
                    panels.append(panel)
                else:
                    # Worker hasn't produced its first result yet → show raw
                    w_elapsed = (time.perf_counter() - worker.start_time
                                 if worker.start_time else 0.0)
                    panels.append(make_waiting_panel(display_frame, name, w_elapsed))

            # ── Build and display canvas ───────────────────────────────────────
            canvas = make_canvas(panels)

            now = time.perf_counter()
            fps_times.append(now)
            fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0]) \
                  if len(fps_times) >= 2 else 0.0

            rec_elapsed = (time.time() - rec_start) if (recording and rec_start) else 0.0
            canvas = draw_status_bar(canvas, recording, rec_elapsed,
                                     ai_driving, show_overlay, fps)

            if take_shot:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUTPUT_DIR / f"beamng_screenshot_{ts}.png"
                cv2.imwrite(str(path), canvas)
                print(f"[Shot] → {path.name}")

            if recording:
                if writer is None:
                    ch, cw = canvas.shape[:2]
                    writer = cv2.VideoWriter(
                        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                        RECORD_FPS, (cw, ch),
                    )
                writer.write(canvas)

            cv2.imshow(win, canvas)
            frame_count += 1

    except KeyboardInterrupt:
        print("\n[Live] Interrupted.")
    finally:
        print("[Live] Shutting down workers …")
        for w in workers.values():
            w.stop()
        for w in workers.values():
            w.join(timeout=5)

        if writer is not None:
            writer.release()
            print(f"[Rec] Saved → {out_path.name}")

        cv2.destroyAllWindows()

        if camera is not None:
            try:
                camera.remove()
            except Exception:
                pass
        if bng is not None:
            try:
                bng.close()
            except Exception:
                pass

        print(f"[Live] Done.  {frame_count} frames processed.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="BeamNG live inference — YOLO26 vs SAM3 (threaded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="both", choices=["both", "yolo26", "sam3"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--prompts", nargs="+", default=None)
    p.add_argument("--yolo-size", default="s", choices=["n", "s", "m", "l", "x"],
                   help="YOLOE-26 size: n/s=fastest, l/x=most accurate  [default: s]")
    p.add_argument("--sam3-weights", default="sam3.pt")
    p.add_argument("--infer-size", type=int, default=640, metavar="W",
                   help="Inference input width in pixels (height auto-scaled). "
                        "Smaller = faster. 640→40+ FPS, 320→80+ FPS on GPU.  [default: 640]")

    p.add_argument("--capture", default="beamngpy", choices=["beamngpy", "screen"])
    p.add_argument("--beamng-home", default=BEAMNG_HOME)
    p.add_argument("--map", default=DEFAULT_MAP)
    p.add_argument("--spawn-idx", type=int, default=0)
    p.add_argument("--vehicle", default=DEFAULT_VEH)

    p.add_argument("--width",  type=int,   default=DEFAULT_W)
    p.add_argument("--height", type=int,   default=DEFAULT_H)
    p.add_argument("--fov",    type=float, default=DEFAULT_FOV)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
