"""
Demo script for off-road lane detection
Supports pretrained encoder models
"""
import os
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt

from model import UNet, UNetSmall, get_model


def load_model(checkpoint_path, arch='unet', encoder='resnet34', device='cuda'):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

    # Create model based on architecture
    if arch == 'unet_basic':
        # Auto-detect features from checkpoint
        features = []
        for i in range(10):
            key = f'downs.{i}.conv.0.weight'
            if key in state_dict:
                features.append(state_dict[key].shape[0])
            else:
                break
        if not features:
            features = [64, 128, 256, 512]
        model = UNet(in_channels=3, out_channels=1, features=features)
    elif arch == 'unet_small':
        features = []
        for i in range(10):
            key = f'downs.{i}.conv.0.weight'
            if key in state_dict:
                features.append(state_dict[key].shape[0])
            else:
                break
        if not features:
            features = [32, 64, 128, 256]
        model = UNetSmall(in_channels=3, out_channels=1, features=features)
    else:
        # Use pretrained encoder model
        model = get_model(
            arch=arch,
            encoder=encoder,
            pretrained=False,  # We'll load weights from checkpoint
            in_channels=3,
            classes=1
        )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded {arch} ({encoder}) from {checkpoint_path}")
    return model


def preprocess_image(image, img_size=512):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def predict(model, image, device, threshold=0.5, hood_mask_ratio=0.0, sky_mask_ratio=0.0):
    """Run prediction on a single image

    Args:
        hood_mask_ratio: Fraction of bottom of image to mask out (0.0-0.5)
                        Use ~0.15-0.2 to hide car hood
        sky_mask_ratio: Fraction of top of image to mask out (0.0-0.5)
                       Use ~0.3-0.4 to exclude sky/distant areas
    """
    with torch.no_grad():
        output = model(image.to(device))
        mask = torch.sigmoid(output)
        mask = (mask > threshold).float()

    mask_np = mask.squeeze().cpu().numpy()

    # Mask out top portion (sky area)
    if sky_mask_ratio > 0:
        h = mask_np.shape[0]
        cutoff = int(h * sky_mask_ratio)
        mask_np[:cutoff, :] = 0

    # Mask out bottom portion (car hood area)
    if hood_mask_ratio > 0:
        h = mask_np.shape[0]
        cutoff = int(h * (1 - hood_mask_ratio))
        mask_np[cutoff:, :] = 0

    return mask_np


def create_overlay(original_image, mask, color=(0, 255, 0), alpha=0.5):
    """Create green overlay on traversable road area"""
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    h, w = original_image.shape[:2]
    mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    mask_resized = (mask_resized > 0.5).astype(np.float32)

    overlay = original_image.copy().astype(np.float32)
    mask_bool = mask_resized > 0.5

    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * original_image[:, :, c] + alpha * color[c],
            original_image[:, :, c]
        )

    return overlay.astype(np.uint8)


def demo_single_image(model, image_path, device, img_size=512, threshold=0.5, output_path=None, hood_mask=0.0, sky_mask=0.0):
    """Run demo on a single image"""
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess_image(original_image, img_size)
    mask = predict(model, input_tensor, device, threshold, hood_mask, sky_mask)
    overlay = create_overlay(original_image, mask, color=(0, 255, 0), alpha=0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Traversable Road Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output_path}")

    plt.show()
    return overlay


def demo_video(model, video_path, device, img_size=512, threshold=0.5, output_path=None, hood_mask=0.0, sky_mask=0.0):
    """Run demo on a video file"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    if hood_mask > 0:
        print(f"Hood mask: bottom {hood_mask*100:.0f}% of image will be ignored")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        input_tensor = preprocess_image(pil_image, img_size)
        mask = predict(model, input_tensor, device, threshold, hood_mask, sky_mask)

        overlay = create_overlay(frame_rgb, mask, color=(0, 255, 0), alpha=0.5)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        if writer:
            writer.write(overlay_bgr)

        cv2.imshow('Off-Road Lane Detection', overlay_bgr)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Video saved to: {output_path}")
    cv2.destroyAllWindows()


def demo_webcam(model, device, img_size=512, threshold=0.5, hood_mask=0.0, sky_mask=0.0):
    """Run demo with webcam"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        input_tensor = preprocess_image(pil_image, img_size)
        mask = predict(model, input_tensor, device, threshold, hood_mask, sky_mask)

        overlay = create_overlay(frame_rgb, mask, color=(0, 255, 0), alpha=0.5)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imshow('Off-Road Lane Detection', overlay_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def demo_dataset(model, data_root, device, split='testing', num_samples=5, img_size=512,
                 threshold=0.5, output_dir=None):
    """Run demo on dataset samples"""
    image_dir = os.path.join(data_root, split, 'image_data')
    gt_dir = os.path.join(data_root, split, 'gt_image')

    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))

    for i, idx in enumerate(indices):
        img_name = images[idx]
        img_path = os.path.join(image_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name.replace('.png', '_fillcolor.png'))

        original_image = Image.open(img_path).convert('RGB')
        gt_arr = np.array(Image.open(gt_path).convert('L'))
        gt_mask = (gt_arr == 255).astype(np.float32)

        input_tensor = preprocess_image(original_image, img_size)
        pred_mask = predict(model, input_tensor, device, threshold)

        pred_overlay = create_overlay(original_image, pred_mask, color=(0, 255, 0), alpha=0.5)
        gt_overlay = create_overlay(original_image, gt_mask, color=(0, 255, 0), alpha=0.5)

        if num_samples == 1:
            ax = axes
        else:
            ax = axes[i]

        ax[0].imshow(original_image)
        ax[0].set_title(f'Original: {img_name[:20]}...')
        ax[0].axis('off')

        ax[1].imshow(gt_overlay)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')

        ax[2].imshow(pred_overlay)
        ax[2].set_title('Predicted')
        ax[2].axis('off')

        pred_resized = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
        diff = np.abs(pred_resized - gt_mask)
        ax[3].imshow(diff, cmap='hot')
        ax[3].set_title('Difference')
        ax[3].axis('off')

    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'demo_results.png'), dpi=150, bbox_inches='tight')
        print(f"Saved to: {os.path.join(output_dir, 'demo_results.png')}")

    plt.show()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, args.arch, args.encoder, device)
    print("Model loaded successfully!")

    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        demo_single_image(model, args.input, device, args.img_size,
                         args.threshold, args.output, args.hood_mask, args.sky_mask)

    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        demo_video(model, args.input, device, args.img_size,
                  args.threshold, args.output, args.hood_mask, args.sky_mask)

    elif args.mode == 'webcam':
        demo_webcam(model, device, args.img_size, args.threshold, args.hood_mask, args.sky_mask)

    elif args.mode == 'dataset':
        demo_dataset(model, args.data_root, device, args.split,
                    args.num_samples, args.img_size, args.threshold, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Off-road lane detection demo')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, default='unet',
                        choices=['unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet', 'unet_basic', 'unet_small'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone (resnet34, resnet50, efficientnet-b0, etc.)')
    parser.add_argument('--mode', type=str, default='dataset',
                        choices=['image', 'video', 'webcam', 'dataset'],
                        help='Demo mode')
    parser.add_argument('--input', type=str, default=None,
                        help='Input image or video path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path')
    parser.add_argument('--data_root', type=str, default='datasets/ORFD',
                        help='Dataset root')
    parser.add_argument('--split', type=str, default='testing',
                        help='Dataset split')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold')
    parser.add_argument('--hood_mask', type=float, default=0.0,
                        help='Mask out bottom X%% of image to hide car hood (e.g., 0.15 for 15%%)')
    parser.add_argument('--sky_mask', type=float, default=0.0,
                        help='Mask out top X%% of image to exclude sky (e.g., 0.3 for 30%%)')

    args = parser.parse_args()
    main(args)
