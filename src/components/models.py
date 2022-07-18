import os
from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from pytorch_lightning import LightningModule
from transformers import AutoModel
import traceback

from .loss_functions import FocalLoss

class FpModelBase(LightningModule, metaclass=ABCMeta):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model = self.create_base_model()
        self.dropout = nn.Dropout(self.config["dropout_rate"])

        self.criterion = eval(config["loss"]["name"])(
            **self.config["loss"]["params"]
        )

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    @abstractmethod
    def create_base_model(self):
        pass

    @abstractmethod
    def forward(self, ids, masks):
        pass

    def training_step(self, batch, batch_idx):
        ids, masks, labels = batch
        logits = self.forward(ids, masks)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "label": label}

    def validation_step(self, batch, batch_idx):
        ids, masks, labels = batch
        logits = self.forward(ids, masks)
        loss = self.criterion(logits, labels)
        logit = logits.detach()
        prob = logits.softmax(axis=1).detach()
        label = labels.detach()
        return {"loss": loss, "logit": logit, "prob": prob, "label": label}

    def predict_step(self, batch, batch_idx):
        ids, masks = batch
        logits = self.forward(ids, masks)
        prob = logits.softmax(axis=1).detach()
        return {"prob": prob}

    def training_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(logits, labels)
        self.min_loss = np.nanmin([self.min_loss, metrics.detach().cpu().numpy()])
        self.log(f"train_loss", metrics)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        logits = torch.cat([out["logit"] for out in outputs])
        probs = torch.cat([out["prob"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(logits, labels)
        self.log(f'val_loss', metrics)

        self.val_probs = probs.detach().cpu().numpy()
        self.val_labels = labels.detach().cpu().numpy()

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(),
            **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer,
            **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

class FpModel(FpModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.fc = self.create_fully_connected()

    def create_base_model(self):
        base_model = AutoModel.from_pretrained(
            self.config["base_model_name"],
            return_dict=False
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, ids, masks):
        out = self.base_model(ids, masks)
        out = self.dropout(out[0][:, 0, :])
        out = self.fc(out)
        return out

class FpModelV1(FpModelBase):
    def __init__(self, config):
        super().__init__(config)

        # const
        self.conv1d_0 = nn.Conv1d(
            **self.config["conv_0"]["params"]
        )
        self.conv1d_1 = nn.Conv1d(
            **self.config["conv_1"]["params"]
        )

    def create_base_model(self):
        base_model = AutoModel.from_pretrained(
            self.config["base_model_name"],
            return_dict=False
        )
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def forward(self, ids, masks):
        out = self.base_model(ids, masks)
        out = self.dropout(out[0])
        out = out.permute(0, 2, 1)
        out = self.conv1d_0(out)
        out = F.relu(out)
        out = self.conv1d_1(out)
        out, _ = torch.max(out, dim=2)
        return out
