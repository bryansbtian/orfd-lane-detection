"""
CARLA Testing Script for UNet Off-Road Lane Detection
Supports: Manual control, Autopilot, and Recording modes

Usage:
    1. Start CARLA 0.10: cd carla10 && CarlaUnreal.exe
    2. Run this script: python carla_test.py --checkpoint checkpoints/unet_20260122_183432/best_model.pth

Controls:
    W/S     - Throttle/Brake
    A/D     - Steer left/right
    SPACE   - Handbrake
    P       - Toggle autopilot
    R       - Toggle recording
    T       - Toggle lane detection overlay
    M       - Change map (cycle through maps)
    N       - Next spawn point (respawn at different location)
    Q/ESC   - Quit
"""

import os
import time
import argparse
import numpy as np
from datetime import datetime

import cv2
import torch
from PIL import Image
from torchvision import transforms

import carla

from model import UNet, UNetSmall, get_model


class LaneDetector:
    """Lane detector for CARLA - supports pretrained encoders"""

    def __init__(self, checkpoint_path, model_type='unet', encoder='resnet34', img_size=512, threshold=0.5, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.threshold = threshold

        # Load model based on type
        if model_type == 'unet_basic':
            self.model = UNet(in_channels=3, out_channels=1)
        elif model_type == 'unet_small':
            self.model = UNetSmall(in_channels=3, out_channels=1)
        else:
            # Use pretrained encoder model
            self.model = get_model(
                arch=model_type,
                encoder=encoder,
                pretrained=False,  # We'll load weights from checkpoint
                in_channels=3,
                classes=1
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"Loaded {model_type} ({encoder}) from {checkpoint_path}")
        print(f"Using device: {self.device}")

    def predict(self, image):
        """Run prediction on a numpy array (H, W, 3) BGR image"""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Preprocess
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            mask = torch.sigmoid(output)
            mask = (mask > self.threshold).float()

        return mask.squeeze().cpu().numpy()

    def create_overlay(self, image, mask, color=(0, 255, 0), alpha=0.5):
        """Create green overlay on detected lane area"""
        h, w = image.shape[:2]
        mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask_resized > 0.5

        overlay = image.copy().astype(np.float32)

        # Apply green overlay where mask is True
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                (1 - alpha) * image[:, :, c] + alpha * color[c],
                image[:, :, c]
            )

        return overlay.astype(np.uint8)


class CarlaTester:
    """CARLA simulation controller with lane detection"""

    def __init__(self, args):
        self.args = args
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.current_frame = None
        self.detector = None
        self.available_maps = []

        # State flags
        self.autopilot = False
        self.recording = False
        self.show_overlay = True
        self.running = True
        self.current_map_idx = 0
        self.current_spawn_idx = 0

        # Recording
        self.video_writer = None
        self.record_start_time = None

        # Control state
        self.control = carla.VehicleControl()

    def connect(self):
        """Connect to CARLA server"""
        print(f"Connecting to CARLA at {self.args.host}:{self.args.port}...")
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        current_map = self.world.get_map().name
        print(f"Connected! Current map: {current_map}")

        # Get available maps
        self.available_maps = [m.split('/')[-1] for m in self.client.get_available_maps()]
        print(f"Available maps: {', '.join(self.available_maps)}")

        # Set synchronous mode for consistent frame capture
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.args.fps
        self.world.apply_settings(settings)

    def load_map(self, map_name):
        """Load a specific map"""
        # Find the full map path from available maps
        full_map_paths = self.client.get_available_maps()
        target_path = None
        for path in full_map_paths:
            if map_name in path:
                target_path = path
                break

        if not target_path:
            print(f"Map '{map_name}' not found. Available: {self.available_maps}")
            return False

        print(f"Loading map: {target_path}... (this may take a while)")
        self.client.set_timeout(120.0)  # Increase timeout for map loading
        self.client.load_world(target_path)
        self.world = self.client.get_world()
        self.client.set_timeout(10.0)  # Reset to normal timeout

        # Re-apply synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.args.fps
        self.world.apply_settings(settings)

        print(f"Map loaded: {map_name}")
        return True

    # Good off-road spawn point indices for each map
    OFFROAD_SPAWNS = {
        'Mine_01': [2, 3, 5, 6, 10, 18, 19],  # Road spawn points at higher elevation
        'Town10HD': [0, 50, 100],
        'Town10HD_Opt': [0, 50, 100],
    }

    def spawn_vehicle(self):
        """Spawn the ego vehicle at an off-road friendly location"""
        blueprint_library = self.world.get_blueprint_library()

        # Get a vehicle blueprint (use off-road capable vehicle)
        vehicle_bp = blueprint_library.filter('vehicle.nissan.patrol')[0]

        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("No spawn points found!")
            return False

        # Get current map name
        current_map = self.world.get_map().name.split('/')[-1].replace('_Opt', '')

        # Use specific spawn point if provided via args
        if hasattr(self.args, 'spawn') and self.args.spawn is not None:
            spawn_idx = min(self.args.spawn, len(spawn_points) - 1)
        # Use predefined off-road spawn points for known maps
        elif current_map in self.OFFROAD_SPAWNS:
            spawn_idx = self.OFFROAD_SPAWNS[current_map][0]
            spawn_idx = min(spawn_idx, len(spawn_points) - 1)
        else:
            spawn_idx = 0

        spawn_point = spawn_points[spawn_idx]
        self.current_spawn_idx = spawn_idx

        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"Spawned vehicle at spawn point {spawn_idx}/{len(spawn_points)}: {spawn_point.location}")
            return True
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            # Try next spawn point
            for i, sp in enumerate(spawn_points):
                if i == spawn_idx:
                    continue
                try:
                    self.vehicle = self.world.spawn_actor(vehicle_bp, sp)
                    print(f"Spawned vehicle at fallback spawn point {i}")
                    return True
                except:
                    continue
            return False

    def setup_camera(self):
        """Attach RGB camera to vehicle"""
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')

        # Configure camera
        camera_bp.set_attribute('image_size_x', str(self.args.width))
        camera_bp.set_attribute('image_size_y', str(self.args.height))
        camera_bp.set_attribute('fov', str(self.args.fov))

        # Camera position (in front of vehicle, road view only)
        camera_transform = carla.Transform(
            carla.Location(x=4.0, z=1.0),  # Well ahead of the vehicle
            carla.Rotation(pitch=-5)  # Slight downward angle to see road
        )

        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.camera.listen(self._on_camera_frame)
        print(f"Camera attached: {self.args.width}x{self.args.height} @ {self.args.fov}° FOV")

    def _on_camera_frame(self, image):
        """Callback for camera frames"""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        self.current_frame = array[:, :, :3]  # BGR

    def load_detector(self):
        """Load the lane detection model"""
        self.detector = LaneDetector(
            checkpoint_path=self.args.checkpoint,
            model_type=self.args.arch,
            encoder=self.args.encoder,
            img_size=self.args.img_size,
            threshold=self.args.threshold
        )

    def process_keyboard(self, key):
        """Process keyboard input"""
        if key == ord('q') or key == 27:  # Q or ESC
            self.running = False

        elif key == ord('p'):  # Toggle autopilot
            self.autopilot = not self.autopilot
            self.vehicle.set_autopilot(self.autopilot)
            print(f"Autopilot: {'ON' if self.autopilot else 'OFF'}")

        elif key == ord('r'):  # Toggle recording
            self.toggle_recording()

        elif key == ord('t'):  # Toggle overlay
            self.show_overlay = not self.show_overlay
            print(f"Lane overlay: {'ON' if self.show_overlay else 'OFF'}")

        elif key == ord('m'):  # Change map
            if self.available_maps:
                self.current_map_idx = (self.current_map_idx + 1) % len(self.available_maps)
                new_map = self.available_maps[self.current_map_idx]
                self.cleanup_actors()
                if self.load_map(new_map):
                    self.spawn_vehicle()
                    self.setup_camera()

        elif key == ord('n'):  # Next spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            self.current_spawn_idx = (self.current_spawn_idx + 1) % len(spawn_points)
            self.cleanup_actors()
            self.args.spawn = self.current_spawn_idx
            self.spawn_vehicle()
            self.setup_camera()
            print(f"Respawned at spawn point {self.current_spawn_idx}/{len(spawn_points)}")

        # Manual controls (when not in autopilot)
        elif not self.autopilot:
            if key == ord('w'):
                self.control.throttle = min(self.control.throttle + 0.1, 1.0)
                self.control.brake = 0.0
            elif key == ord('s'):
                self.control.brake = min(self.control.brake + 0.1, 1.0)
                self.control.throttle = 0.0
            elif key == ord('a'):
                self.control.steer = max(self.control.steer - 0.1, -1.0)
            elif key == ord('d'):
                self.control.steer = min(self.control.steer + 0.1, 1.0)
            elif key == ord(' '):  # Spacebar - handbrake
                self.control.hand_brake = not self.control.hand_brake

    def toggle_recording(self):
        """Toggle video recording"""
        if self.recording:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            print("Recording stopped")
        else:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.args.output_dir, f"carla_recording_{timestamp}.mp4")
            os.makedirs(self.args.output_dir, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.args.fps,
                (self.args.width, self.args.height)
            )
            self.recording = True
            self.record_start_time = time.time()
            print(f"Recording started: {output_path}")

    def create_hud(self, frame):
        """Add HUD overlay with status information"""
        h, w = frame.shape[:2]

        # Status text
        speed = self.vehicle.get_velocity()
        speed_kmh = 3.6 * np.sqrt(speed.x**2 + speed.y**2 + speed.z**2)

        lines = [
            f"Speed: {speed_kmh:.1f} km/h",
            f"Autopilot: {'ON' if self.autopilot else 'OFF'}",
            f"Recording: {'ON' if self.recording else 'OFF'}",
            f"Overlay: {'ON' if self.show_overlay else 'OFF'}",
            f"Map: {self.world.get_map().name.split('/')[-1]}",
        ]

        if self.recording and self.record_start_time:
            elapsed = time.time() - self.record_start_time
            lines.append(f"Rec time: {elapsed:.1f}s")

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 30 + 25 * len(lines)), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # Draw text
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (20, 35 + 25 * i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Controls help
        help_lines = [
            "W/S=Throttle/Brake, A/D=Steer, SPACE=Handbrake",
            "P=Autopilot, R=Record, T=Overlay, M=Map, N=Respawn, Q=Quit"
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(frame, line, (20, h - 40 + 20 * i),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def run(self):
        """Main loop"""
        print("\n" + "="*50)
        print("CARLA Lane Detection Test")
        print("="*50)

        # Connect and setup
        self.connect()

        # Load specific map if requested
        if self.args.map:
            current_map = self.world.get_map().name.split('/')[-1]
            if self.args.map not in current_map:
                self.load_map(self.args.map)

        # Find current map index for cycling
        current_map = self.world.get_map().name.split('/')[-1]
        for i, m in enumerate(self.available_maps):
            if m in current_map or current_map in m:
                self.current_map_idx = i
                break

        # Spawn vehicle and camera
        if not self.spawn_vehicle():
            print("Failed to spawn vehicle. Exiting.")
            return
        self.setup_camera()

        # Load lane detector
        self.load_detector()

        print("\nReady! Use keyboard to control the vehicle.")
        print("Press 'P' for autopilot, 'R' to record, 'Q' to quit\n")

        cv2.namedWindow('CARLA Lane Detection', cv2.WINDOW_NORMAL)

        try:
            while self.running:
                # Tick the simulation
                self.world.tick()

                # Apply manual control if not in autopilot
                if not self.autopilot:
                    # Gradually decay steering and throttle
                    self.control.steer *= 0.9
                    if abs(self.control.steer) < 0.01:
                        self.control.steer = 0.0
                    self.vehicle.apply_control(self.control)

                # Process camera frame
                if self.current_frame is not None:
                    display_frame = self.current_frame.copy()

                    # Run lane detection
                    if self.show_overlay:
                        mask = self.detector.predict(self.current_frame)
                        display_frame = self.detector.create_overlay(
                            display_frame, mask, color=(0, 255, 0), alpha=0.4
                        )

                    # Add HUD
                    display_frame = self.create_hud(display_frame)

                    # Record if enabled
                    if self.recording and self.video_writer:
                        self.video_writer.write(display_frame)

                    # Display
                    cv2.imshow('CARLA Lane Detection', display_frame)

                # Process keyboard
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    self.process_keyboard(key)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup_actors(self):
        """Destroy spawned actors"""
        if self.camera:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

    def cleanup(self):
        """Cleanup on exit"""
        print("\nCleaning up...")

        # Stop recording
        if self.video_writer:
            self.video_writer.release()

        # Destroy actors
        self.cleanup_actors()

        # Reset world settings
        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        cv2.destroyAllWindows()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Test UNet lane detection on CARLA')

    # CARLA connection
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--map', type=str, default=None,
                        help='Map to load (default: Town07 for off-road)')
    parser.add_argument('--spawn', type=int, default=None,
                        help='Spawn point index (default: uses predefined off-road locations)')

    # Camera settings
    parser.add_argument('--width', type=int, default=1280,
                        help='Camera width')
    parser.add_argument('--height', type=int, default=720,
                        help='Camera height')
    parser.add_argument('--fov', type=int, default=120,
                        help='Camera field of view')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS')

    # Model settings
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet', 'unet_basic', 'unet_small'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (resnet34, resnet50, efficientnet-b0, etc.)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Model input size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold')

    # Output
    parser.add_argument('--output_dir', type=str, default='carla_recordings',
                        help='Directory for recordings')

    args = parser.parse_args()

    tester = CarlaTester(args)
    tester.run()


if __name__ == '__main__':
    main()
