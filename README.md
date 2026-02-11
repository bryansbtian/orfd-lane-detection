# Off-Road Lane Detection

A deep learning system for detecting traversable road areas in off-road environments using semantic segmentation with pretrained encoders. The model identifies drivable paths and overlays them with a green highlight for visualization.

## Features

- **Multiple architectures**: U-Net, U-Net++, DeepLabV3+, FPN, PSPNet
- **Pretrained encoders**: ResNet, EfficientNet, MobileNet, and more via ImageNet weights
- **Strong data augmentation**: Rain, fog, sun flare, shadows, color jitter using albumentations
- **Real-time inference** on images, videos, webcam feeds, and CARLA simulator
- **CARLA 0.10 integration** for testing in simulation
- **Training pipeline** with mixed precision, cosine annealing, and focal loss

## Project Structure

```
Off Road/
├── train.py             # Training script
├── model.py             # Model architectures and loss functions
├── preprocessing.py     # Dataset loaders with augmentation
├── demo.py              # Demo/inference script (image, video, webcam)
├── lane_detector.py     # Simulator-agnostic lane detection wrapper
├── carla_inference.py   # CARLA simulator testing
├── requirements.txt     # Python dependencies
├── .gitignore
├── README.md
├── carla/               # CARLA 0.10 simulator
├── carla916/            # CARLA 0.9.16 simulator
├── checkpoints/         # Saved model checkpoints
├── datasets/
│   ├── ORFD/            # ORFD dataset
│   │   ├── training/
│   │   │   ├── image_data/
│   │   │   └── gt_image/
│   │   ├── validation/
│   │   └── testing/
│   └── RELLIS/          # Rellis-3D dataset
│       ├── raw/
│       └── labeled/
├── test_videos/         # Input test videos
└── output_videos/       # Inference output videos
```

## Model Options

### Architectures

| Architecture | Description                                    |
| ------------ | ---------------------------------------------- |
| `unet`       | Classic U-Net with skip connections            |
| `unetpp`     | U-Net++ with nested skip connections           |
| `deeplabv3p` | DeepLabV3+ with ASPP and decoder (recommended) |
| `fpn`        | Feature Pyramid Network                        |
| `pspnet`     | Pyramid Scene Parsing Network                  |

### Encoders (Backbones)

| Encoder           | Parameters | Speed   | Accuracy |
| ----------------- | ---------- | ------- | -------- |
| `resnet34`        | 24M        | Fast    | Good     |
| `resnet50`        | 26M        | Medium  | Better   |
| `efficientnet-b0` | 5M         | Fast    | Good     |
| `efficientnet-b3` | 12M        | Medium  | Better   |
| `mobilenet_v2`    | 3.5M       | Fastest | Decent   |

## Installation

### Requirements

- Python 3.12
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/bryansbtian/orfd-lane-detection.git
cd "Off Road"

# Create conda environment
conda create -n offroad python=3.12 -y
conda activate offroad

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

Extract the datasets into the `datasets/` folder:

```bash
# ORFD dataset
unzip ORFD.zip -d datasets/

# RELLIS-3D dataset
unzip RELLIS.zip -d datasets/
```

This will create the following structure:

```
datasets/
├── ORFD/
│   ├── training/
│   │   ├── image_data/     # RGB images (*.png)
│   │   └── gt_image/       # Ground truth masks (*_fillcolor.png)
│   ├── validation/
│   │   └── ...
│   └── testing/
│       └── ...
└── RELLIS/
    ├── raw/                # RGB images organized by sequence
    └── labeled/            # Label masks organized by sequence
```

## Usage

### Training

```bash
# DeepLabV3+ with ResNet50 on ORFD and Rellis-3D datasets
python train.py --arch deeplabv3p --encoder resnet50 --orfd_root datasets/ORFD --rellis_root datasets/RELLIS --img_size 512 --amp

# U-Net with ResNet34 (best balance of speed and accuracy)
python train.py --arch unet --encoder resnet34 --img_size 512 --amp

# Lightweight model for edge deployment
python train.py --arch unet --encoder efficientnet-b0 --img_size 512 --amp

# Original U-Net without pretrained encoder (not recommended)
python train.py --arch unet_basic --img_size 256
```

#### Training Options

| Parameter         | Type  | Default         | Description                                                                                            |
| ----------------- | ----- | --------------- | ------------------------------------------------------------------------------------------------------ |
| `--arch`          | str   | `unet`          | Architecture: `unet`, `unetpp`, `deeplabv3`, `deeplabv3p`, `fpn`, `pspnet`, `unet_basic`, `unet_small` |
| `--encoder`       | str   | `resnet34`      | Encoder backbone (see table above)                                                                     |
| `--no_pretrained` | flag  | False           | Don't use ImageNet pretrained weights                                                                  |
| `--img_size`      | int   | 512             | Input image size                                                                                       |
| `--batch_size`    | int   | 8               | Batch size                                                                                             |
| `--epochs`        | int   | 100             | Number of epochs                                                                                       |
| `--lr`            | float | 1e-4            | Learning rate                                                                                          |
| `--patience`      | int   | 20              | Early stopping patience                                                                                |
| `--loss`          | str   | `focal_dice`    | Loss function: `focal_dice` or `combined`                                                              |
| `--scheduler`     | str   | `cosine`        | LR scheduler: `cosine` or `plateau`                                                                    |
| `--amp`           | flag  | False           | Enable mixed precision training                                                                        |
| `--resume`        | str   | None            | Resume from checkpoint                                                                                 |
| `--data_root`     | str   | `datasets/ORFD` | Dataset root directory                                                                                 |

Training creates a timestamped folder in `checkpoints/` containing:

- `best_model.pth` - Model with best validation IoU
- `latest_model.pth` - Most recent checkpoint
- `training_history.png` - Loss and metrics plots
- `results.txt` - Final metrics summary

### Demo / Inference

```bash
# Run on AA2 test video
python demo.py --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth --arch deeplabv3p --encoder resnet50 --mode video --input test_videos/aa2.mp4 --output output_videos/aa2_overlay.mp4

# Run on test dataset samples
python demo.py --checkpoint checkpoints/unet_resnet34_XXXXX/best_model.pth --arch unet --encoder resnet34

# Run on single image
python demo.py --checkpoint path/to/model.pth --arch unet --encoder resnet34 --mode image --input photo.jpg --output result.png

# Run on video
python demo.py --checkpoint path/to/model.pth --arch unet --encoder resnet34 --mode video --input video.mp4 --output output.mp4

# Run with webcam
python demo.py --checkpoint path/to/model.pth --arch unet --encoder resnet34 --mode webcam
```

### CARLA Simulator Testing

Test your model in the CARLA simulator. The script can automatically launch the simulator if it's not running.

**Supported Versions:**

- CARLA 0.10 (`carla/`)
- CARLA 0.9.16 (`carla916/`)

```bash
# Run with CARLA 0.10 (default)
conda activate YOUR_CARLA_0.10.0_ENVIRONMENT
python carla_inference.py --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth --arch deeplabv3p --encoder resnet50 --map Mine_01

# Run with CARLA 0.9.16
conda activate YOUR_CARLA_0.9.16_ENVIRONMENT
python carla_inference.py --carla_version 0.9.16 --checkpoint checkpoints/deeplabv3p_resnet50_20260204_180353/best_model.pth --arch deeplabv3p --encoder resnet50 --map Town10HD
```

The script will automatically find the CARLA installation in the corresponding folder (`carla` or `carla916`) and configure the Python path. If the simulator is not running, it will be started in the background.

#### CARLA Controls

| Key   | Action              |
| ----- | ------------------- |
| W/S   | Throttle / Brake    |
| A/D   | Steer left / right  |
| SPACE | Handbrake           |
| P     | Toggle autopilot    |
| R     | Toggle recording    |
| T     | Toggle lane overlay |
| M     | Cycle maps          |
| N     | Next spawn point    |
| Q/ESC | Quit                |

#### CARLA Options

| Parameter         | Type             | Default      | Description                               |
| ----------------- | ---------------- | ------------ | ----------------------------------------- |
| `--checkpoint`    | str              | **required** | Path to model checkpoint                  |
| `--carla_version` | `0.10`, `0.9.16` | `0.10`       | Version of the CARLA simulator to use     |
| `--arch`          | str              | `unet`       | Model architecture                        |
| `--encoder`       | str              | `resnet34`   | Encoder backbone                          |
| `--map`           | str              | None         | Map to load (e.g., `Mine_01`, `Town10HD`) |
| `--img_size`      | int              | 512          | Model input size                          |
| `--threshold`     | float            | 0.5          | Prediction threshold                      |
| `--fov`           | int              | 120          | Camera field of view                      |

## Data Augmentation

The training pipeline includes strong augmentation via albumentations:

- Horizontal flip
- Shift, scale, rotate
- Gaussian noise, blur, motion blur
- Brightness, contrast, hue/saturation changes
- Random rain, fog, sun flare
- Random shadows
- CLAHE (adaptive histogram equalization)
- Cutout / coarse dropout

## Tips

### Memory Issues

- Use smaller encoder: `efficientnet-b0` or `mobilenet_v2`
- Reduce `--batch_size` (try 4 or 2)
- Reduce `--img_size` (try 384 or 256)
- Enable `--amp` for mixed precision

### Better Results

- Use pretrained encoders (default) - significantly improves performance
- Use `--img_size 512` or higher for finer details
- Try `deeplabv3p` architecture for best segmentation quality
- Lower `--threshold` (e.g., 0.3) for more sensitive detection
- Collect domain-specific data and fine-tune

## Metrics

- **IoU (Intersection over Union)**: Measures overlap between prediction and ground truth
- **Dice Score**: Similar to IoU, emphasizes overlap
- **Focal + Dice Loss**: Handles class imbalance better than BCE
