# models/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
import json
import os

class StandartClassifier(nn.Module):
    def __init__(self, num_classes=50, pretrained=False, training_layers=3):
        super().__init__()
        self.backbone = models.efficientnet_b3(pretrained=pretrained)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        # Freeze earlier layers
        children = list(self.backbone.children())
        for child in children[:-training_layers]:
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, x):
        return F.log_softmax(self.backbone(x), dim=1)


class BirdModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = StandartClassifier(
            num_classes=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            training_layers=config["model"]["training_layers"]
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        acc = (y_pred.argmax(dim=1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = F.nll_loss(y_pred, y)
        acc = (y_pred.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["train"]["lr"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=self.hparams["train"]["lr_gamma"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": self.hparams["train"]["lr_step_every"],
                "monitor": "val_acc"
            }
        }

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_directory, "model.bin"))
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.hparams, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory, map_location=None):
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(config)
        weight_path = os.path.join(save_directory, "model.bin")
        state_dict = torch.load(weight_path, map_location=map_location)
        model.model.load_state_dict(state_dict)
        return model