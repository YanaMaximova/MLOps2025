import os
import json
import numpy as np
from skimage.io import imsave
import pytest

from src.dataset import BirdDataset




def create_mock_dataset(tmp_path):
    img_dir = tmp_path / "train_images"
    img_dir.mkdir()

    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    imsave(img_dir / "bird1.jpg", img)
    imsave(img_dir / "bird2.jpg", img)

    gt = {"bird1.jpg": 0, "bird2.jpg": 1}
    gt_path = tmp_path / "train_gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f)

    return str(img_dir), str(gt_path), gt


def test_gt_format_and_structure(tmp_path):
    _, gt_path, gt = create_mock_dataset(tmp_path)

    with open(gt_path) as f:
        loaded_gt = json.load(f)

    assert isinstance(loaded_gt, dict)
    for fname, label in loaded_gt.items():
        assert isinstance(fname, str)
        assert fname.lower().endswith(('.jpg', '.jpeg', '.png'))
        assert isinstance(label, int)
        assert 0 <= label < 50


def test_dataset_output_structure(tmp_path):
    img_dir, gt_path, gt = create_mock_dataset(tmp_path)

    ds = BirdDataset(
        mode="any",
        gt=gt,
        img_dir=img_dir,
    )
    x, y = ds[0]

    assert hasattr(x, 'numpy')
    x_np = x.numpy()
    assert x_np.shape == (3, 350, 350), f"Expected (3,350,350), got {x_np.shape}"
    assert x_np.dtype == np.float32
    assert -3.0 < x_np.min() < 0.0 < x_np.max() < 3.0

    assert isinstance(y, int)
    assert 0 <= y < 50

def test_dataset_missing_file_error(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    
    gt = {"missing.jpg": 0}
    
    ds = BirdDataset(mode="any", gt=gt, img_dir=str(img_dir))
    
    with pytest.raises(FileNotFoundError):
        _ = ds[0]