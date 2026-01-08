import os
import json
import torch
import pytest

from predict import classify, compute_accuracy


@pytest.fixture
def mock_config(tmp_path):
    config = {
        "model": {"input_size": 350, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "inference": {"batch_size": 1, "num_workers": 0}
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        json.dump(config, f)
    return str(config_path)


@pytest.fixture
def mock_gt(tmp_path):
    gt = {"img1.jpg": 0}
    gt_path = tmp_path / "gt.json"
    with open(gt_path, "w") as f:
        json.dump(gt, f)
    return str(gt_path)


class MockDataset:
    def __init__(self, *args, **kwargs):
        self.filenames = ["img1.jpg"]
    def __len__(self): return 1
    def __getitem__(self, _):
        return torch.zeros(3, 350, 350), "img1.jpg"

class MockDataLoader:
    def __init__(self, *args, **kwargs): pass
    def __iter__(self):
        return iter([(torch.zeros(1, 3, 350, 350), ["img1.jpg"])])


class MockBirdModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return torch.tensor([[2.1, -0.3, 1.8]])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def test_postprocess_logits_to_class_id(mock_config, monkeypatch):
    monkeypatch.setattr("predict.BirdModel", MockBirdModel)
    monkeypatch.setattr("predict.InferenceDataset", MockDataset)
    monkeypatch.setattr("predict.DataLoader", MockDataLoader)

    preds = classify(
        model_path="any",
        test_img_dir="any",
        config_path=mock_config, 
        batch_size=1
    )

    assert preds == {"img1.jpg": 0}


def test_compute_accuracy_label_mismatch(mock_gt, tmp_path):
    preds = {"img1.jpg": 50}
    acc = compute_accuracy(preds, mock_gt) 
    assert acc == 0.0

def test_classify_empty_test_directory(monkeypatch, mock_config):
    class EmptyDataset:
        def __init__(self, *args, **kwargs): self.filenames = []
        def __len__(self): return 0
        def __getitem__(self, _): raise IndexError

    class EmptyDataLoader:
        def __init__(self, *args, **kwargs): pass
        def __iter__(self): return iter([])

    monkeypatch.setattr("predict.BirdModel", MockBirdModel)
    monkeypatch.setattr("predict.InferenceDataset", EmptyDataset)
    monkeypatch.setattr("predict.DataLoader", EmptyDataLoader)

    preds = classify(
        model_path="any",
        test_img_dir="empty",
        config_path=mock_config
    )
    assert preds == {}


def test_compute_accuracy_missing_files(mock_gt):
    preds = {"img1.jpg": 0, "unknown.jpg": 1}
    acc = compute_accuracy(preds, mock_gt)
    assert acc == 1.0


def test_compute_accuracy_empty_gt(mock_gt, tmp_path):
    empty_gt = {}
    empty_gt_path = tmp_path / "empty_gt.json"
    with open(empty_gt_path, "w") as f:
        json.dump(empty_gt, f)

    preds = {"img1.jpg": 0}
    acc = compute_accuracy(preds, str(empty_gt_path))
    assert acc is None


def test_classify_nonexistent_test_dir(monkeypatch, mock_config):
    monkeypatch.setattr("predict.BirdModel", MockBirdModel)

    with pytest.raises(FileNotFoundError):
        classify(
            model_path="any",
            test_img_dir="/non/existent/path",
            config_path=mock_config
        )