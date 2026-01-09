import json
import random
import os
from utils.config import load_config


def main():
    config = load_config('config.yaml')
    random.seed(config['data']['seed'])
    os.makedirs(config['data']['prepare_dir'], exist_ok=True)

    with open(config['data']['train_gt_path']) as f:
        gt = json.load(f)

    class_to_imgs = {}
    for img, cls in gt.items():
        class_to_imgs.setdefault(cls, []).append(img)

    train_gt, val_gt = {}, {}

    for cls, imgs in class_to_imgs.items():
        random.shuffle(imgs)
        split = int(len(imgs) * config['data']['fraction'])
        for i in imgs[:split]:
            train_gt[i] = cls
        for i in imgs[split:]:
            val_gt[i] = cls

    with open(f"{config['data']['prepare_dir']}/train_gt.json", "w") as f:
        json.dump(train_gt, f, indent=2)

    with open(f"{config['data']['prepare_dir']}/val_gt.json", "w") as f:
        json.dump(val_gt, f, indent=2)

    print("Prepare stage done")

if __name__ == "__main__":
    main()