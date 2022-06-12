import os
import sys
import glob
import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.special import softmax
import torch
import torch.optim as optim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel

config = {
    "mode": "train",
    "epoch": 1,
    "n_splits": 2,
    "random_seed": 57,
    "label": "discourse_effectiveness",
    "experiment_name": "roberta-v0",
    "path": {
        "traindata": "/kaggle/input/feedback-prize-effectiveness/train.csv",
        "testdata": "/kaggle/input/feedback-prize-effectiveness/test.csv",
        "temporal_dir": "../tmp/artifacts/"
    }
}
config["model"] = {
    "base_model_name": "/kaggle/input/roberta-base",
    "dim_feature": 768,
    "num_class": 3,
    "optimizer":{
        "name": "optim.AdamW",
        "params":{
            "lr": 1e-5
        },
    },
    "scheduler":{
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params":{
            "T_0": 20,
            "eta_min": 1e-4,
        }
    }
}
config["earlystopping"] = {
    'patience': 5
}
config["checkpoint"] = {
    "dirpath": "../tmp/artifacts/",
    "monitor": "val_loss",
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "gpus": 1,
    "accumulate_grad_batches": 1,
    "fast_dev_run": False,
    "num_sanity_val_steps": 0,
    "resume_from_checkpoint": None,
}
config["datamodule"] = {
    "dataset":{
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "max_length": 128,
        "discourse_type": {
            "Claim": 0,
            "Concluding Statement": 1,
            "Counterclaim": 2,
            "Evidence": 3,
            "Lead": 4,
            "Position": 5,
            "Rebuttal": 6
        },
        "discourse_effectiveness": {
            "Adequate": 0,
            "Effective": 1,
            "Ineffective": 2
        },
    },
    "train_loader": {
        "batch_size": 64,
        "shuffle": True,
        "num_workers": 4,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": False
    },
    "test_loader": {
        "batch_size": 64,
        "shuffle": False,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": False
    }
}

transforms = {
    "train": T.Compose([
        T.ToTensor()
    ]),
    "valid": T.Compose([
        T.ToTensor()
    ]),
    "pred": T.Compose([
        T.ToTensor()
    ])
}

class PeDataset(Dataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.val = df["discourse_text"].values
        self.label = None
        if "discourse_effectiveness" in df.keys():
            self.labels = F.one_hot(
                torch.tensor(
                    [config["discourse_effectiveness"][d] for d in df["discourse_effectiveness"]]
                ),
                num_classes=self.config["num_class"]
            ).float()
        self.tokenizer = Tokenizer.from_pretrained(config["base_model_name"])
        self.transform = transform

    def __len__(self):
        return len(self.val)

    def __getitem__(self, idx):
        ids, masks = self.tokenize(self.val[idx])
        if self.transform is not None:
            ids = self.transform(ids)
        if self.labels is not None:
            label = self.labels[idx]
            return ids, masks, label
        return ids, masks

    def tokenize(self, text):
        token = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding="max_length"
        )
        ids = torch.tensor(token["input_ids"], dtype=torch.long)
        masks = torch.tensor(token["attention_mask"], dtype=torch.long)
        return ids, masks


class PeDataModule(LightningDataModule):
    def __init__(
        self,
        df_train,
        df_val,
        df_pred,
        Dataset,
        Tokenizer,
        config,
        transforms
    ):
        super().__init__()

        # const
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.df_pred = df_pred
        self.transforms = transforms

        # class
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer

    def train_dataloader(self):
        dataset = self.Dataset(self.df_train, self.config["dataset"], self.Tokenizer)
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = self.Dataset(self.df_val, self.config["dataset"], self.Tokenizer)
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = self.Dataset(self.df_pred, self.config["dataset"], self.Tokenizer)
        return DataLoader(dataset, **self.config["pred_loader"])


class PeModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model = self.create_model()
        self.fc = self.create_fully_connected()

        self.criterion = nn.BCEWithLogitsLoss()

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def create_model(self):
        return AutoModel.from_pretrained(self.config["base_model_name"], return_dict=False)

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, ids, masks):
        out = self.base_model(ids, masks)
        out = self.fc(out[0][:, 0])
        return out

    def training_step(self, batch, batch_idx):
        ids, masks, labels = batch
        logits = self.forward(ids, masks)
        loss = self.criterion(logits, labels)
        prob = logits.sigmoid().detach().cpu()
        label = labels.detach().cpu()
        return {"loss": loss, "prob": prob, "label": label}

    def validation_step(self, batch, batch_idx):
        ids, masks, labels = batch
        logits = self.forward(ids, masks)
        loss = self.criterion(logits, labels)
        prob = logits.sigmoid().detach().cpu()
        label = labels.detach().cpu()
        return {"loss": loss, "prob": prob, "label": label}

    def predict_step(self, batch, batch_idx):
        ids, masks = batch
        logits = self.forward(ids, masks)
        prob = logits.sigmoid().detach().cpu()
        return {"prob": prob}

    def training_epoch_end(self, outputs):
        probs = torch.cat([out["prob"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(probs, labels)
        self.min_loss = np.nanmin([self.min_loss, metrics.detach().cpu().numpy()])
        self.log(f"train_loss", metrics)
        return super().training_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        probs = torch.cat([out["prob"] for out in outputs])
        labels = torch.cat([out["label"] for out in outputs])
        metrics = self.criterion(probs, labels)
        self.log(f'val_loss', metrics)

        self.val_probs = probs.detach().cpu().numpy()
        self.val_labels = labels.detach().cpu().numpy()

        return super().validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer,
            **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

class Trainer:
    def __init__(self, Model, DataModule, Dataset, Tokenizer, df_train, config, transforms, mlflow_logger):
        # const
        self.mlflow_logger = mlflow_logger
        self.config = config
        self.df_train = df_train
        self.transforms = transforms
        self.skf = StratifiedKFold(
            self.config["n_splits"],
            shuffle=True,
            random_state=self.config["random_seed"])

        # variable
        self.min_loss = np.nan
        self.val_probs = []
        self.val_labels = []

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer

    def run(self):
        list_val_probs = []
        list_val_labels = []
        for fold, (idx_train, idx_val) in enumerate(self.skf.split(self.df_train, self.df_train[self.config["label"]])):
            # create datamodule
            datamodule = self._create_datamodule(idx_train, idx_val)

            # train
            min_loss = self._train_with_crossvalid(datamodule, fold)
            self.min_loss = np.nanmin([self.min_loss, min_loss])

            # valid
            val_probs, val_labels = self._valid(datamodule, fold)
            list_val_probs.append(val_probs)
            list_val_labels.append(val_labels)
        self.val_probs = np.concatenate(list_val_probs)
        self.val_labels =np.concatenate(list_val_labels)

    def _create_datamodule(self, idx_train, idx_val):
        df_train_fold = self.df_train.loc[idx_train].reset_index(drop=True)
        df_val_fold = self.df_train.loc[idx_val].reset_index(drop=True)
        datamodule = self.DataModule(
            df_train=df_train_fold,
            df_val=df_val_fold,
            df_pred=None,
            Dataset=self.Dataset,
            Tokenizer=self.Tokenizer,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _train_with_crossvalid(self, datamodule, fold):
        model = self.Model(self.config["model"])
        checkpoint_name = f"best_loss_{fold}"

        earystopping = EarlyStopping(
            monitor="val_loss",
            **self.config["earlystopping"]
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            **self.config["checkpoint"]
        )

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            max_epochs=self.config["epoch"],
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt"
        )

        min_loss = model.min_loss
        return min_loss

    def _train_without_valid(self, datamodule, min_loss):
        model = self.Model(self.config["model"])
        checkpoint_name = f"best_loss"

        earystopping = EarlyStopping(
            monitor="train_loss",
            stopping_threshold=min_loss,
            **self.config["earlystopping"]
        )
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            filename=checkpoint_name,
            **self.config["checkpoint"]
        )

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            max_epochs=self.config["epoch"],
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            f"{self.config['path']['temporal_dir']}/{checkpoint_name}.ckpt"
        )

    def _valid(self, datamodule, fold):
        checkpoint_name = f"best_loss_{fold}"
        model = self.Model.load_from_checkpoint(
            f"{config['path']['temporal_dir']}/{checkpoint_name}.ckpt",
            config=self.config["model"]
        )
        model.eval()

        trainer = pl.Trainer(
            logger=self.mlflow_logger,
            max_epochs=self.config["epoch"],
            **self.config["trainer"]
        )

        trainer.validate(model, datamodule=datamodule)

        val_probs = model.val_probs
        val_labels = model.val_labels
        return val_probs, val_labels


def create_mlflow_logger(config):
    if not (config["mode"]=="train"):
        return None
    timestamp = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y/%m/%d %H:%M:%S"
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=timestamp
    )
    return mlflow_logger


if __name__=="__main__":

    mlflow_logger = create_mlflow_logger(config)

    df_train = pd.read_csv(config["path"]["traindata"]).iloc[:128]
    df_test = pd.read_csv(config["path"]["testdata"])

    trainer = Trainer(
        PeModel,
        PeDataModule,
        PeDataset,
        AutoTokenizer,
        df_train,
        config,
        transforms,
        mlflow_logger
    )
    trainer.run()

    print(trainer.val_probs)
    print(trainer.val_labels)
