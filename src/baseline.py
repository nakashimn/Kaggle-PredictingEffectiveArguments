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
    "path": {
        "traindata": "/kaggle/input/feedback-prize-effectiveness/train.csv",
        "testdata": "/kaggle/input/feedback-prize-effectiveness/test.csv",
        "temporal_dir": "../tmp/artifacts/"
    },
    "base_model_name": "/kaggle/input/roberta-base",
    "experiment_name": "roberta-v0",
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
    "max_length": 128,
    "dim_feature": 768,
    "num_class": 3,
    "epoch": 100,
    "model_name": "best_loss",
    "n_splits": 5,
    "random_seed": 57,
    "trainer": {
        "gpus": 1,
        "accumulate_grad_batches": 1,
        "fast_dev_run": False,
        "num_sanity_val_steps": 0,
        "resume_from_checkpoint": None,
    },
    "train_loader": {
        "batch_size": 2,
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
    },
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

class Dataset(Dataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.val = df["discourse_text"].values
        self.label = None
        if "discourse_effectiveness" in df.keys():
            self.labels = F.one_hot(
                torch.tensor(
                    [config["discourse_effectiveness"][d] for d in df["discourse_effectiveness"]]
                ), num_classes=self.config["num_class"]
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


class DataModule(LightningDataModule):
    def __init__(
        self,
        df_train,
        df_val,
        df_pred,
        Tokenizer,
        config,
        transforms
    ):
        super().__init__()
        self.config = config
        self.df_train = df_train
        self.df_val = df_val
        self.df_pred = df_pred
        self.Tokenizer = Tokenizer
        self.transforms = transforms

    def train_dataloader(self):
        dataset = Dataset(self.df_train, self.config, self.Tokenizer)
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = Dataset(self.df_val, self.config, self.Tokenizer)
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = Dataset(self.df_pred, self.config, self.Tokenizer)
        return DataLoader(dataset, **self.config["pred_loader"])


class Model(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.base_model = self.create_model()
        self.fc = self.create_fully_connected()

        self.criterion = nn.BCEWithLogitsLoss() # loss function

        self.val_probs = np.nan
        self.val_preds = np.nan
        self.val_labels = np.nan

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



if __name__=="__main__":

    # Logger
    timestamp = datetime.datetime.strftime(
        datetime.datetime.now(), "%Y/%m/%d %H:%M:%S"
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=config["experiment_name"],
        run_name=timestamp
    )

    filepath_train_csv = config["path"]["traindata"]
    filepath_test_csv = config["path"]["testdata"]

    df_train = pd.read_csv(filepath_train_csv)
    df_test = pd.read_csv(filepath_test_csv)

    skf = StratifiedKFold(config["n_splits"], shuffle=True, random_state=config["random_seed"])
    for fold, (idx_train, idx_val) in enumerate(skf.split(df_train, df_train["discourse_effectiveness"])):
        df_train_fold = df_train.loc[idx_train].reset_index(drop=True)
        df_val_fold = df_train.loc[idx_val].reset_index(drop=True)
        datamodule = DataModule(
            df_train_fold,
            df_val_fold,
            df_test,
            AutoTokenizer,
            config,
            transforms
        )

        checkpoint_name = f"best_loss_fold{fold}"

        # train
        model = Model(config)

        earystopping = EarlyStopping(monitor="val_loss")
        lr_monitor = callbacks.LearningRateMonitor()
        loss_checkpoint = callbacks.ModelCheckpoint(
            dirpath=config["path"]["temporal_dir"],
            filename=checkpoint_name,
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            save_weights_only=False
        )

        trainer = pl.Trainer(
            logger=mlflow_logger,
            max_epochs=config["epoch"],
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            f"{config['path']['temporal_dir']}/{checkpoint_name}.ckpt"
        )

        # valid
        model = Model.load_from_checkpoint(
            f"{config['path']['temporal_dir']}/{checkpoint_name}.ckpt",
            config=config
        )
        model.eval()

        trainer.validate(model, datamodule=datamodule)



    # output
    # df_preds = pd.DataFrame(list_preds)

    # df_preds.to_csv("submission.csv", index=None)
