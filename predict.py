port argparse
import os
import json
import numpy as np
import cv2
import albumentations as A
from skimage.io import imread
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.model import BirdModel
from utils.config import load_config

class InferenceDataset(Dataset):
    def __init__(self, img_dir, config):
        self.config = config
        self.img_dir = img_dir
        self.filenames = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        self.transform = self._default_transform()

    def _default_transform(self):
        size = self.config["model"]["input_size"]
        mean = self.config["model"]["mean"]
        std = self.config["model"]["std"]
        return A.Compose([
            A.CenterCrop(width=size, height=size),
            A.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.img_dir, fname)
        image = imread(path).astype(np.float32)

        h, w = image.shape[:2]
        size = self.config["model"]["input_size"]
        if h < w:
            h_new, w_new = size, int(size * w / h)
        else:
            h_new, w_new = int(size * h / w), size
        image = cv2.resize(image, (w_new, h_new))

        if image.ndim == 2:
            image = np.dstack([image] * 3)

        image = self.transform(image=image)['image']
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image, fname

def classify(model_path, test_img_dir, config_path="config.yaml", batch_size=None):
    config = load_config(config_path)
    model = BirdModel.from_pretrained(model_path, map_location="cpu")
    model.eval()

    dataset = InferenceDataset(test_img_dir, config)
    batch_size = batch_size or config["inference"]["batch_size"]
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["inference"]["num_workers"],
        pin_memory=False,
        persistent_workers=True
    )

    predictions = {}
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch, filenames in tqdm(dataloader, desc="Predicting"):
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            for fname, pred in zip(filenames, preds):
                predictions[fname] = int(pred)

    return predictions

def compute_accuracy(predictions, gt_path):
    if not gt_path:
        return None
    with open(gt_path, "r") as f:
        gt = json.load(f)

    y_true, y_pred = [], []
    for fname, pred in predictions.items():
        if fname in gt:
            y_true.append(gt[fname])
            y_pred.append(pred)

    if len(y_true) == 0:
        return None

    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    print(f" Accuracy: {accuracy:.4f} ({sum(1 for t, p in zip(y_true, y_pred) if t == p)}/{len(y_true)})")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_dir", required=False)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--batch_size", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    test_dir = args.test_dir or config["data"]["test_img_dir"]
    model = args.model or config["save"]["final_model_path"]

    preds = classify(
        model_path=model,
        test_img_dir=test_dir,
        config_path=args.config,
        batch_size=args.batch_size
    )

    with open(args.output, "w") as f:
        json.dump(preds, f, indent=2)
    compute_accuracy(preds, config["data"]["test_gt_path"])
    print(f"Saved {len(preds)} predictions to {args.output}")