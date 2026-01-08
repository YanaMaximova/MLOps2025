import argparse
import json
import torch
import os
import random
import numpy as np
from pathlib import Path

from utils.config import load_config
from utils.logging_setup import setup_logging
from src.dataset import BirdDataset
from src.model import BirdModel
from torch.utils.data import DataLoader
import albumentations as A
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(config_path: str, verbose: bool = False):
    config = load_config(config_path)
    logger = setup_logging(config)
    logger.info("Starting training")

    set_seed(config["train"]["seed"])

    with open(config["data"]["train_gt_path"], "r") as f:
        train_gt = json.load(f)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
    ])

    ds_train = BirdDataset(
        mode="train", gt=train_gt, img_dir=config["data"]["train_img_dir"],
        fraction=config["data"]["fraction"], transform=train_transform
    )
    ds_val = BirdDataset(
        mode="val", gt=train_gt, img_dir=config["data"]["train_img_dir"],
        fraction=config["data"]["fraction"]
    )

    dl_train = DataLoader(ds_train, batch_size=config["train"]["batch_size"], shuffle=True, num_workers=4)
    dl_val = DataLoader(ds_val, batch_size=config["train"]["val_batch_size"], shuffle=False, num_workers=4)

    model = BirdModel(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config["save"]["checkpoint_dir"],
        filename="best-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor="val_acc",
        patience=config["train"]["early_stopping_patience"],
        verbose=True,
        mode="max"
    )


    if torch.backends.mps.is_available():
        accelerator = "mps"
        precision = "32"
    elif torch.cuda.is_available():
        accelerator = "cuda"
        precision = "16-mixed"
    else:
        accelerator = "cpu"
        precision = "32"

    trainer = pl.Trainer(
        max_epochs=config["train"]["max_epochs"],
        accelerator=accelerator, 
        devices=1,
        precision=precision, 
        logger=pl.loggers.CSVLogger("logs", name="birds"),
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True,
        enable_progress_bar=verbose,
    )


    logger.info("Training started...")
    trainer.fit(model, dl_train, dl_val)


    final_path = config["save"]["final_model_path"]
    logger.info(f"Saving final model to {final_path}")
    model.save_pretrained(final_path)

    if hasattr(trainer, "callback_metrics"):
        val_acc = trainer.callback_metrics.get("val_acc", "N/A")
        val_loss = trainer.callback_metrics.get("val_loss", "N/A")
        print(f"Final validation metrics → val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")

    logger.info("Training completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    main(args.config, args.verbose)