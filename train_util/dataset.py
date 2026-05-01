import sys
sys.path.append('./')

import numpy as np
import random
import cv2
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset


class BasicImageDataset(Dataset):
    """Reads degraded image folders that each contain original.png, Target*.png, hints, captions, and a .pth feature file."""

    def __init__(self, dataset_paths, is_train=True):
        self.image_files = []
        self.path = dataset_paths
        self.is_train = is_train
        assert isinstance(dataset_paths, list)
        for _ds in dataset_paths:
            if self.is_train:
                self.image_files.extend(Path(_ds).glob(f"**/*original.png"))
            else:
                self.image_files.extend(list(Path(_ds).glob(f"**/*original.png"))[:32])

    def __len__(self):
        return len(self.image_files) if self.is_train else 32 * len(self.path)

    def _read_images(self, path):
        images = sorted(Path(path).glob(f"**/*.png"))
        target = original = None
        hints = []
        for i in images:
            if "Target" in str(i.name):
                target = np.array(Image.open(i))[..., :3]
            elif "original" in str(i.name):
                original = np.array(Image.open(i))[..., :3]
            else:
                hints.append(np.array(Image.open(i))[..., :3])
        return target, original, np.concatenate(hints, axis=-1)

    def _read_captions(self, path):
        captions = sorted(Path(path).glob(f"**/*.txt"))
        cps = {}
        for c in captions:
            text = open(c).readline()
            if '1_5' in str(c.name):
                cps['1_5'] = text
            elif '2_1' in str(c.name):
                cps['2_1'] = text
        return cps

    def _read_features(self, path):
        pth_files = sorted(Path(path).glob(f"**/*.pth"))
        return torch.load(pth_files[0], weights_only=False)

    def getitem(self, idx):
        folder = self.image_files[idx].parent
        target, original, hints = self._read_images(folder)
        captions = self._read_captions(folder)
        features = self._read_features(folder)
        return captions, target, original, hints, features, folder


class AIRDataset(Dataset):
    """Wraps BasicImageDataset with cropping, normalisation, and prompt dropout."""

    def __init__(self, task_dataset, task_id, task_name, hints=True,
                 train_unconditional_guidance=True, sd_model='1_5'):
        self.task_dataset = task_dataset
        self.task_id = task_id
        self.task_name = task_name
        self.hints = hints
        self.train_unconditional_guidance = train_unconditional_guidance
        self.sd_model = sd_model

    def __len__(self):
        return len(self.task_dataset)

    def _resize_image(self, image, resolution=512):
        """Random-crop to square then resize. Returns image and relative crop coords."""
        H, W, C = image.shape
        if W >= H:
            crop = H
            crop_l = random.randint(0, W - crop)
            crop_r = crop_l + crop
            crop_t, crop_b = 0, H
        else:
            crop = W
            crop_t = random.randint(0, H - crop)
            crop_b = crop_t + crop
            crop_l, crop_r = 0, W
        image = image[crop_t:crop_b, crop_l:crop_r]
        fH, fW = float(H), float(W)
        k = resolution / min(fH, fW)
        image = cv2.resize(image, (resolution, resolution),
                           interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return image, [crop_t / fH, crop_b / fH, crop_l / fW, crop_r / fW]

    def _crop_to_sizes(self, image, sizes, resolution=512):
        """Apply the same relative crop to a second image then resize."""
        H, W, C = image.shape
        ct, cb, cl, cr = int(sizes[0] * H), int(sizes[1] * H), int(sizes[2] * W), int(sizes[3] * W)
        image = image[ct:cb, cl:cr]
        fH, fW = float(H), float(W)
        k = resolution / min(fH, fW)
        image = cv2.resize(image, (resolution, resolution),
                           interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
        return image

    def __getitem__(self, idx):
        assert idx < len(self.task_dataset)
        captions, target_img, source_img, other_hints, task_features, _ = self.task_dataset.getitem(idx % len(self.task_dataset))

        if self.hints:
            all_hints = np.concatenate([source_img / 255.0, other_hints / 255.0], axis=-1)
        else:
            all_hints = source_img / 255.0

        source_img_crop, sizes = self._resize_image(source_img)
        all_hints = self._crop_to_sizes(all_hints, sizes)
        target_crop = self._crop_to_sizes(target_img[..., :3], sizes)

        # Normalize target to [-1, 1]
        target_crop = (target_crop.astype(np.float32) / 127.5) - 1.0

        # Text prompt with dropout
        if self.sd_model not in captions:
            prompt = list(captions.values())[0] if captions else ''
        else:
            prompt = captions[self.sd_model]
        if self.train_unconditional_guidance:
            prompt = prompt if random.uniform(0, 1) > 0.4 else ''
        else:
            prompt = ''

        task_id = np.array(self.task_id).astype(np.float32)

        return dict(
            jpg=target_crop.astype(np.float32),
            txt=prompt,
            hint=all_hints.astype(np.float32),
            task_id=task_id,
            task_features=task_features,
            task_name=self.task_name,
        )
