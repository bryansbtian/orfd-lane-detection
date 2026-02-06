"""
Segmentation models for off-road lane detection
Uses pretrained encoders for much better performance
"""
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("WARNING: segmentation-models-pytorch not installed. Using basic UNet only.")
    print("Install with: pip install segmentation-models-pytorch")


def get_model(arch='unet', encoder='resnet34', pretrained=True, in_channels=3, classes=1):
    """
    Get a segmentation model with pretrained encoder

    Args:
        arch: Architecture - 'unet', 'unetpp', 'deeplabv3', 'deeplabv3p', 'fpn', 'pspnet'
        encoder: Encoder backbone - 'resnet34', 'resnet50', 'efficientnet-b0', 'efficientnet-b3',
                 'mobilenet_v2', 'timm-efficientnet-b0', etc.
        pretrained: Use ImageNet pretrained weights
        in_channels: Input channels (3 for RGB)
        classes: Output classes (1 for binary segmentation)

    Returns:
        PyTorch model
    """
    if not SMP_AVAILABLE:
        print("Falling back to basic UNet")
        return UNet(in_channels=in_channels, out_channels=classes)

    weights = "imagenet" if pretrained else None

    if arch == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == 'deeplabv3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == 'deeplabv3p':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )
    elif arch == 'pspnet':
        model = smp.PSPNet(
            encoder_name=encoder,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


class ConvBlock(nn.Module):
    """Double convolution block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """Original U-Net architecture (no pretrained encoder)"""

    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class UNetSmall(nn.Module):
    """Smaller U-Net for faster training"""

    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss"""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


class CombinedFocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss - better for imbalanced data"""

    def __init__(self, focal_weight=0.5, dice_weight=0.5, alpha=0.25, gamma=2.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.focal_weight * self.focal(pred, target) + self.dice_weight * self.dice(pred, target)


def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 1.0

    return (intersection / union).item()


def calculate_dice(pred, target, threshold=0.5):
    """Calculate Dice coefficient"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = target.float()

    intersection = (pred * target).sum()

    if pred.sum() + target.sum() == 0:
        return 1.0

    return (2. * intersection / (pred.sum() + target.sum())).item()


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)

    if SMP_AVAILABLE:
        model = get_model(arch='unet', encoder='resnet34', pretrained=True)
        out = model(x)
        print(f"ResNet34-UNet output shape: {out.shape}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"ResNet34-UNet parameters: {total_params:,}")

        for arch in ['deeplabv3p', 'fpn']:
            model = get_model(arch=arch, encoder='resnet34', pretrained=True)
            out = model(x)
            params = sum(p.numel() for p in model.parameters())
            print(f"{arch}-ResNet34 params: {params:,}")

    model = UNet(in_channels=3, out_channels=1)
    out = model(x)
    print(f"\nOriginal UNet output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Original UNet parameters: {total_params:,}")