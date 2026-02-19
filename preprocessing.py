"""
Dataset loaders for off-road lane detection
Supports ORFD and Rellis-3D datasets
With improved augmentation using albumentations
"""
import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import numpy as np
from torchvision import transforms
import random

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print("WARNING: albumentations not installed. Using basic augmentation.")
    print("Install with: pip install albumentations")


def get_training_augmentation(img_size=256):
    """Strong augmentation for training"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.1), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=(3, 7), p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.5),
        A.OneOf([
            A.RandomRain(drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=5, brightness_coefficient=0.9, p=1),
            A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=1),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=1),
        ], p=0.2),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=5, p=0.3),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_augmentation(img_size=256):
    """Minimal augmentation for validation/test"""
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class ORFDDataset(Dataset):
    """Off-Road Freespace Detection Dataset with strong augmentation"""

    def __init__(self, root_dir, split='training', img_size=(256, 256), augment=False):
        """
        Args:
            root_dir: Path to ORFD dataset root
            split: 'training', 'validation', or 'testing'
            img_size: Target image size (height, width) - can be int or tuple
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.augment = augment

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.image_dir = os.path.join(root_dir, split, 'image_data')
        self.mask_dir = os.path.join(root_dir, split, 'gt_image')

        self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.png')])

        if ALBUMENTATIONS_AVAILABLE:
            if augment:
                self.transform = get_training_augmentation(self.img_size[0])
            else:
                self.transform = get_validation_augmentation(self.img_size[0])
        else:
            self.transform = None
            self.img_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))

        mask_name = img_name.replace('.png', '_fillcolor.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = np.array(Image.open(mask_path).convert('L'))

        mask = (mask == 255).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            else:
                mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

            if self.augment:
                image_pil, mask_pil = self._basic_augment(image_pil, mask_pil)

            mask_pil = mask_pil.resize(self.img_size, Image.NEAREST)
            image = self.img_transform(image_pil)

            mask_arr = np.array(mask_pil)
            mask_arr = (mask_arr > 127).astype(np.float32)
            mask = torch.from_numpy(mask_arr).float().unsqueeze(0)

        return image, mask

    def _basic_augment(self, image, mask):
        """Basic augmentation fallback"""
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(np.random.uniform(0.7, 1.3))

        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(np.random.uniform(0.7, 1.3))

        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(np.random.uniform(0.7, 1.3))

        return image, mask

    def get_original_image(self, idx):
        """Get original image without normalization for visualization"""
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        return image, img_name


# Rellis-3D class mapping for traversability
# Based on official Rellis-3D dataset documentation
RELLIS_TRAVERSABLE_CLASSES = [
    0,   # void/background - often represents traversable ground
    1,   # dirt
    3,   # grass
    10,  # asphalt
    23,  # concrete
    29,  # puddle
    30,  # mud
    34,  # dirt (alternate)
]


class Rellis3DDataset(Dataset):
    """
    Rellis-3D Dataset for off-road traversability detection
    Converts multi-class labels to binary (traversable/non-traversable)
    """

    def __init__(self, root_dir, split='training', img_size=(256, 256), augment=False,
                 traversable_classes=None, split_ratio=(0.7, 0.15, 0.15), seed=42):
        """
        Args:
            root_dir: Path to Rellis dataset root (containing 'raw' and 'labeled' folders)
            split: 'training', 'validation', or 'testing'
            img_size: Target image size (height, width) - can be int or tuple
            augment: Whether to apply data augmentation
            traversable_classes: List of class IDs to consider as traversable (default: RELLIS_TRAVERSABLE_CLASSES)
            split_ratio: (train, val, test) ratios for splitting the dataset
            seed: Random seed for reproducible splits
        """
        self.root_dir = root_dir
        self.split = split
        self.augment = augment
        self.traversable_classes = traversable_classes or RELLIS_TRAVERSABLE_CLASSES

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.raw_dir = os.path.join(root_dir, 'raw')
        self.labeled_dir = os.path.join(root_dir, 'labeled')

        # Collect all image-label pairs across sequences
        self.samples = []
        sequences = sorted([d for d in os.listdir(self.labeled_dir)
                           if os.path.isdir(os.path.join(self.labeled_dir, d))])

        for seq in sequences:
            label_folder = os.path.join(self.labeled_dir, seq, 'pylon_camera_node_label_id')
            raw_folder = os.path.join(self.raw_dir, seq, 'pylon_camera_node')

            if not os.path.exists(label_folder) or not os.path.exists(raw_folder):
                continue

            for label_file in sorted(os.listdir(label_folder)):
                if not label_file.endswith('.png'):
                    continue
                # Match raw image (.jpg) with label (.png)
                raw_file = label_file.replace('.png', '.jpg')
                raw_path = os.path.join(raw_folder, raw_file)
                label_path = os.path.join(label_folder, label_file)

                if os.path.exists(raw_path):
                    self.samples.append((raw_path, label_path))

        # Split dataset
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        if split == 'training':
            selected_indices = indices[:n_train]
        elif split == 'validation':
            selected_indices = indices[n_train:n_train + n_val]
        else:  # testing
            selected_indices = indices[n_train + n_val:]

        self.samples = [self.samples[i] for i in selected_indices]

        # Setup transforms
        if ALBUMENTATIONS_AVAILABLE:
            if augment:
                self.transform = get_training_augmentation(self.img_size[0])
            else:
                self.transform = get_validation_augmentation(self.img_size[0])
        else:
            self.transform = None
            self.img_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask_raw = np.array(Image.open(mask_path))

        # Convert multi-class to binary (traversable = 1, non-traversable = 0)
        mask = np.isin(mask_raw, self.traversable_classes).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            else:
                mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

            if self.augment:
                image_pil, mask_pil = self._basic_augment(image_pil, mask_pil)

            mask_pil = mask_pil.resize(self.img_size, Image.NEAREST)
            image = self.img_transform(image_pil)

            mask_arr = np.array(mask_pil)
            mask_arr = (mask_arr > 127).astype(np.float32)
            mask = torch.from_numpy(mask_arr).float().unsqueeze(0)

        return image, mask

    def _basic_augment(self, image, mask):
        """Basic augmentation fallback"""
        if np.random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(np.random.uniform(0.7, 1.3))

        if np.random.random() > 0.5:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(np.random.uniform(0.7, 1.3))

        return image, mask

    def get_original_image(self, idx):
        """Get original image without normalization for visualization"""
        img_path, _ = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return image, os.path.basename(img_path)


# RUGD dataset color-to-class mapping
# Based on RUGD: Robot Unstructured Ground Dataset (Wigness et al., 2019)
RUGD_CLASS_NAMES = [
    "void", "dirt", "sand", "grass", "tree", "pole", "water", "sky",
    "vehicle", "container", "asphalt", "gravel", "building", "mulch",
    "rock-bed", "log", "bicycle", "person", "fence", "bush", "sign",
    "rock", "bridge", "concrete", "picnic-table",
]

# RGB colors from RUGD_annotation-colormap.txt
RUGD_PALETTE = [
    (0, 0, 0),          # 0:  void
    (108, 64, 20),       # 1:  dirt
    (255, 229, 204),     # 2:  sand
    (0, 102, 0),         # 3:  grass
    (0, 255, 0),         # 4:  tree
    (0, 153, 153),       # 5:  pole
    (0, 128, 255),       # 6:  water
    (0, 0, 255),         # 7:  sky
    (255, 255, 0),       # 8:  vehicle
    (255, 0, 127),       # 9:  container/generic-object
    (64, 64, 64),        # 10: asphalt
    (255, 128, 0),       # 11: gravel
    (255, 0, 0),         # 12: building
    (153, 76, 0),        # 13: mulch
    (102, 102, 0),       # 14: rock-bed
    (102, 0, 0),         # 15: log
    (0, 255, 128),       # 16: bicycle
    (204, 153, 255),     # 17: person
    (102, 0, 204),       # 18: fence
    (255, 153, 204),     # 19: bush
    (0, 102, 102),       # 20: sign
    (153, 204, 255),     # 21: rock
    (102, 255, 255),     # 22: bridge
    (101, 101, 11),      # 23: concrete
    (114, 85, 47),       # 24: picnic-table
]

# Default traversable classes for RUGD
RUGD_TRAVERSABLE_CLASSES = [1, 2, 3, 10, 11]  # dirt, sand, grass, asphalt, gravel


def _build_rugd_color_lut(palette):
    """Build a fast (256**3,) lookup table from RGB color -> class ID."""
    lut = np.zeros(256 * 256 * 256, dtype=np.uint8)
    for class_id, (r, g, b) in enumerate(palette):
        lut[r * 65536 + g * 256 + b] = class_id
    return lut


class RUGDDataset(Dataset):
    """
    RUGD Dataset for off-road traversability detection.
    Converts color-coded annotations to binary (traversable / non-traversable).

    Expected layout:
        rugd_root/
            RUGD_frames-with-annotations/   (or 'images/')
                creek/  park-1/  ...
            RUGD_annotations/               (or 'annotations/')
                creek/  park-1/  ...
    """

    def __init__(self, root_dir, split='training', img_size=(256, 256), augment=False,
                 traversable_classes=None, split_ratio=(0.7, 0.15, 0.15), seed=42):
        self.root_dir = root_dir
        self.split = split
        self.augment = augment
        self.traversable_classes = traversable_classes or RUGD_TRAVERSABLE_CLASSES

        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        # Locate image and annotation directories (support common naming conventions)
        self.image_dir = self._find_dir(root_dir, ['RUGD_frames-with-annotations', 'raw', 'images', 'rgb'])
        self.annot_dir = self._find_dir(root_dir, ['RUGD_annotations', 'labeled', 'annotations', 'labels'])

        if self.image_dir is None or self.annot_dir is None:
            raise FileNotFoundError(
                f"Could not find image/annotation dirs in {root_dir}. "
                "Expected RUGD_frames-with-annotations/ and RUGD_annotations/")

        # Build color -> class LUT
        self._color_lut = _build_rugd_color_lut(RUGD_PALETTE)

        # Collect samples across all scene folders
        self.samples = []
        scenes = sorted([d for d in os.listdir(self.annot_dir)
                         if os.path.isdir(os.path.join(self.annot_dir, d))])
        for scene in scenes:
            annot_scene = os.path.join(self.annot_dir, scene)
            image_scene = os.path.join(self.image_dir, scene)
            if not os.path.isdir(image_scene):
                continue
            for fname in sorted(os.listdir(annot_scene)):
                if not fname.endswith('.png'):
                    continue
                img_path = os.path.join(image_scene, fname)
                ann_path = os.path.join(annot_scene, fname)
                if os.path.exists(img_path):
                    self.samples.append((img_path, ann_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No RUGD samples found in {root_dir}")

        # Deterministic train/val/test split
        rng = random.Random(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        if split == 'training':
            sel = indices[:n_train]
        elif split == 'validation':
            sel = indices[n_train:n_train + n_val]
        else:
            sel = indices[n_train + n_val:]
        self.samples = [self.samples[i] for i in sel]

        # Transforms
        if ALBUMENTATIONS_AVAILABLE:
            self.transform = get_training_augmentation(self.img_size[0]) if augment else get_validation_augmentation(self.img_size[0])
        else:
            self.transform = None
            self.img_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        trav_names = [RUGD_CLASS_NAMES[i] for i in self.traversable_classes]
        print(f"RUGD traversable classes: {trav_names}")

    @staticmethod
    def _find_dir(root, candidates):
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def _color_to_classid(self, annot_rgb):
        """Convert (H, W, 3) color annotation to (H, W) class-ID map using LUT."""
        r, g, b = annot_rgb[:, :, 0], annot_rgb[:, :, 1], annot_rgb[:, :, 2]
        flat_idx = r.astype(np.int32) * 65536 + g.astype(np.int32) * 256 + b.astype(np.int32)
        return self._color_lut[flat_idx]

    def __getitem__(self, idx):
        img_path, ann_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        annot = np.array(Image.open(ann_path).convert('RGB'))

        class_map = self._color_to_classid(annot)
        mask = np.isin(class_map, self.traversable_classes).astype(np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).float()
            else:
                mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image_pil = Image.fromarray(image)
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            if self.augment:
                if np.random.random() > 0.5:
                    image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.resize(self.img_size, Image.NEAREST)
            image = self.img_transform(image_pil)
            mask_arr = np.array(mask_pil)
            mask_arr = (mask_arr > 127).astype(np.float32)
            mask = torch.from_numpy(mask_arr).float().unsqueeze(0)

        return image, mask

    def get_original_image(self, idx):
        img_path, _ = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        return image, os.path.basename(img_path)


def get_combined_dataset(orfd_root, rellis_root, split='training', img_size=256, augment=False, rugd_root=None):
    """Create a combined dataset from ORFD, Rellis-3D, and/or RUGD"""
    datasets = []

    if orfd_root and os.path.exists(orfd_root):
        orfd_dataset = ORFDDataset(orfd_root, split=split, img_size=img_size, augment=augment)
        datasets.append(orfd_dataset)
        print(f"ORFD {split}: {len(orfd_dataset)} samples")

    if rellis_root and os.path.exists(rellis_root):
        rellis_dataset = Rellis3DDataset(rellis_root, split=split, img_size=img_size, augment=augment)
        datasets.append(rellis_dataset)
        print(f"Rellis-3D {split}: {len(rellis_dataset)} samples")

    if rugd_root and os.path.exists(rugd_root):
        rugd_dataset = RUGDDataset(rugd_root, split=split, img_size=img_size, augment=augment)
        datasets.append(rugd_dataset)
        print(f"RUGD {split}: {len(rugd_dataset)} samples")

    if len(datasets) == 0:
        raise ValueError("No valid dataset paths provided")
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)


def get_dataloaders(data_root, batch_size=8, img_size=256, num_workers=4):
    """Create train, validation, and test dataloaders"""

    train_dataset = ORFDDataset(data_root, split='training', img_size=img_size, augment=True)
    val_dataset = ORFDDataset(data_root, split='validation', img_size=img_size, augment=False)
    test_dataset = ORFDDataset(data_root, split='testing', img_size=img_size, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_combined_dataloaders(orfd_root=None, rellis_root=None, rugd_root=None, batch_size=8, img_size=256, num_workers=4):
    """Create train, validation, and test dataloaders from combined datasets"""

    train_dataset = get_combined_dataset(orfd_root, rellis_root, split='training', img_size=img_size, augment=True, rugd_root=rugd_root)
    val_dataset = get_combined_dataset(orfd_root, rellis_root, split='validation', img_size=img_size, augment=False, rugd_root=rugd_root)
    test_dataset = get_combined_dataset(orfd_root, rellis_root, split='testing', img_size=img_size, augment=False, rugd_root=rugd_root)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("Testing ORFD Dataset:")
    print("-" * 40)
    orfd_root = 'datasets/ORFD'
    if os.path.exists(orfd_root):
        dataset = ORFDDataset(orfd_root, split='training', augment=True)
        print(f"ORFD Dataset size: {len(dataset)}")
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")

    print("\nTesting Rellis-3D Dataset:")
    print("-" * 40)
    rellis_root = 'datasets/RELLIS'
    if os.path.exists(rellis_root):
        dataset = Rellis3DDataset(rellis_root, split='training', augment=True)
        print(f"Rellis-3D Dataset size: {len(dataset)}")
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")

    print("\nTesting Combined Dataset:")
    print("-" * 40)
    if os.path.exists(orfd_root) and os.path.exists(rellis_root):
        combined = get_combined_dataset(orfd_root, rellis_root, split='training', augment=True)
        print(f"Combined Dataset size: {len(combined)}")

    print(f"\nAlbumentations available: {ALBUMENTATIONS_AVAILABLE}")