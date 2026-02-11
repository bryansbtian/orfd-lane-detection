"""
Simulator-agnostic lane detector.
Takes BGR numpy arrays and returns binary masks.
"""

import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms

from model import UNet, UNetSmall, get_model


class LaneDetector:
    """Lane detector - supports pretrained encoders"""

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

        # Inference (use FP16 on CUDA for speed)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
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
