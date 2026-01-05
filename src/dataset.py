import os
import json
import numpy as np
import cv2
import albumentations as A
from skimage.io import imread
from torch.utils.data import Dataset
import torch

class BirdDataset(Dataset):
    def __init__(self, mode, gt, img_dir, fraction=0.8, transform=None):
        self._items = []
        self._labels = []
        self._transform = transform
        self._normalize = A.Compose([
            A.CenterCrop(width=350, height=350),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        class_to_images = {}
        for img_name, cls in gt.items():
            class_to_images.setdefault(cls, []).append(img_name)

        split = int(fraction * len(class_to_images[next(iter(class_to_images))]))
        if mode == "train":
            selected_files = [img for imgs in class_to_images.values() for img in imgs[:split]]
        elif mode == "val":
            selected_files = [img for imgs in class_to_images.values() for img in imgs[split:]]
        else:
            selected_files = [img for imgs in class_to_images.values() for img in imgs[:2]]

        for img_file in selected_files:
            self._items.append(os.path.join(img_dir, img_file))
            self._labels.append(gt[img_file])

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        item, label = self._items[index], self._labels[index]
        image = imread(item).astype(np.float32)

        h, w = image.shape[:2]
        if h < w:
            h_new, w_new = 350, int(350 * w / h)
        else:
            h_new, w_new = int(350 * h / w), 350
        image = cv2.resize(image, (w_new, h_new))

        if image.ndim == 2:
            image = np.dstack([image] * 3)

        image = self._normalize(image=image)['image']
        if self._transform:
            image = self._transform(image=image)['image']

        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image, label