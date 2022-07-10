import os
import shutil
import sys
import pathlib
import glob
import datetime
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

config = {
    "mode": "train",
    "n_splits": 3,
    "random_seed": 57,
    "label": "discourse_effectiveness",
    "labels": [
        "Ineffective",
        "Adequate",
        "Effective"
    ],
    "types": [
        "Claim",
        "Concluding Statement",
        "Counterclaim",
        "Evidence",
        "Lead",
        "Position",
        "Rebuttal"
    ],
    "experiment_name": "roberta-v0",
    "path": {
        "traindata": "/kaggle/input/feedback-prize-effectiveness/train.csv",
        "trainessay": "/kaggle/input/feedback-prize-effectiveness/train/",
        "testdata": "/kaggle/input/feedback-prize-effectiveness/test.csv",
        "testessay": "/kaggle/input/feedback-prize-effectiveness/test/",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/model/pe-roberta-v0/"
    },
    "modelname": "best_loss",
    "pred_ensemble": False
}
config["model"] = {
    "base_model_name": "/kaggle/input/roberta-base",
    "dim_feature": 768,
    "num_class": 3,
    "freeze_base_model": False,
    "loss":{
        "name": "FocalLoss",
        "params":{
            "gamma": 2.0
        }
    },
    "optimizer":{
        "name": "optim.RAdam",
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
    "dirpath": config["path"]["temporal_dir"],
    "monitor": "val_loss",
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 100,
    "accumulate_grad_batches": 1,
    "fast_dev_run": False,
    "num_sanity_val_steps": 0,
    "resume_from_checkpoint": None,
    "precision": 32
}
config["datamodule"] = {
    "dataset":{
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "use_fast_tokenizer": True,
        "max_length": 512,
        "discourse_effectiveness": {l : i for i, l in enumerate(config["labels"])},
        "discourse_type": {tp : i for i, tp in enumerate(config["types"])}
    },
    "train_loader": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": True,
    },
    "val_loader": {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": True,
        "drop_last": False
    },
    "pred_loader": {
        "batch_size": 8,
        "shuffle": False,
        "num_workers": 16,
        "pin_memory": False,
        "drop_last": False
    }
}
config["Metrics"] = {
    "label": config["labels"]
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


def fetchEssay(essay_id, dirpath):
    """
    Read the text file of the specific essay_id
    """
    essay_path = os.path.join(dirpath, essay_id + '.txt')
    essay_text = open(essay_path, 'r').read()
    return essay_text

class TextCleaner:
    def __init__(self):
        nltk.download('stopwords')
        self.stop = stopwords.words('english')
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(self, text):
        return [self.lemmatizer.lemmatize(w) for w in text]

    def clean(self, data, col):
        # Replace Upper to Lower
        data[col] = data[col].str.lower()
        # Replace
        data[col] = data[col].str.replace(r"what's", "what is ", regex=True)
        data[col] = data[col].str.replace(r"\'ve", " have ", regex=True)
        data[col] = data[col].str.replace(r"can't", "cannot ", regex=True)
        data[col] = data[col].str.replace(r"n't", " not ", regex=True)
        data[col] = data[col].str.replace(r"i'm", "i am ", regex=True)
        data[col] = data[col].str.replace(r"\'re", " are ", regex=True)
        data[col] = data[col].str.replace(r"\'d", " would ", regex=True)
        data[col] = data[col].str.replace(r"\'ll", " will ", regex=True)
        data[col] = data[col].str.replace(r"\'scuse", " excuse ", regex=True)
        data[col] = data[col].str.replace(r"\'s", " ", regex=True)
        # Remove
        data[col] = data[col].str.replace(r'\s', ' ', regex=True)
        data[col] = data[col].str.replace('.', ' ', regex=True)
        data[col] = data[col].str.replace(',', ' ', regex=True)
        data[col] = data[col].str.replace('\"', ' ', regex=True)
        data[col] = data[col].str.replace('(', ' ', regex=True)
        data[col] = data[col].str.replace(')', ' ', regex=True)
        data[col] = data[col].str.replace(':', ' ', regex=True)
        data[col] = data[col].str.replace(';', ' ', regex=True)
        # Clean some punctutations
        data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3', regex=True)
        # Replace repeating characters more than 3 times to length of 3
        data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1', regex=True)
        # Add space around repeating characters
        data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ', regex=True)
        # patterns with repeating characters
        data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1', regex=True)
        data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1', regex=True)
        data[col] = data[col].str.replace(r'[ ]{2,}',' ', regex=True).str.strip()
        data[col] = data[col].str.replace(r'[ ]{2,}',' ', regex=True).str.strip()
        data[col] = data[col].str.replace(r' +', ' ', regex=True)
        # data[col] = data[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stop)]))
        return data

class PeDataset(Dataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.val = (
            df["discourse_type"]+ " " + df["discourse_text"] + " " + df["essay"]
        ).values
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = F.one_hot(
                torch.tensor(
                    [self.config[self.config["label"]][d] for d in df[self.config["label"]]]
                ),
                num_classes=self.config["num_class"]
            ).float()
        self.tokenizer = Tokenizer.from_pretrained(config["base_model_name"], use_fast=self.config["use_fast_tokenizer"])
        self.transform = transform

    def __len__(self):
        return len(self.val)

    def __getitem__(self, idx):
        ids, masks = self.tokenize(self.val[idx])
        if self.transform is not None:
            ids = self.transform(ids)
        if self.labels is not None:
            labels = self.labels[idx]
            return ids, masks, labels
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
        self.transforms = self.read_transforms(transforms)

        # class
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer

    def read_transforms(self, transforms):
        if transforms is not None:
            return transforms
        return {"train": None, "valid": None, "pred": None}

    def train_dataloader(self):
        dataset = self.Dataset(self.df_train, self.config["dataset"], self.Tokenizer, self.transforms["train"])
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = self.Dataset(self.df_val, self.config["dataset"], self.Tokenizer, self.transforms["valid"])
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = self.Dataset(self.df_pred, self.config["dataset"], self.Tokenizer, self.transforms["pred"])
        return DataLoader(dataset, **self.config["pred_loader"])

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred, target):
        probas = pred.softmax(dim=1)
        loss = -(target*((1-probas)**self.gamma)*(probas.log())).sum()
        return loss

class PeModel(LightningModule):
    def __init__(self, config):
        super().__init__()

        # const
        self.config = config
        self.base_model = self.create_model()
        self.fc = self.create_fully_connected()

        self.criterion = eval(config["loss"]["name"])(
            **self.config["loss"]["params"]
        )

        # variables
        self.val_probs = np.nan
        self.val_labels = np.nan
        self.min_loss = np.nan

    def create_model(self):
        base_model = AutoModel.from_pretrained(self.config["base_model_name"], return_dict=False)
        if not self.config["freeze_base_model"]:
            return base_model
        for param in base_model.parameters():
            param.requires_grad = False
        return base_model

    def create_fully_connected(self):
        return nn.Linear(self.config["dim_feature"], self.config["num_class"])

    def forward(self, ids, masks):
        out = self.base_model(ids, masks)
        out = self.fc(out[0][:, 0, :])
        return out

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
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer,
            **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

class MinLoss:
    def __init__(self):
        self.value = np.nan

    def update(self, min_loss):
        self.value = np.nanmin([self.value, min_loss])

class ValidResult:
    def __init__(self):
        self.values = None

    def append(self, values):
        if self.values is None:
            self.values = values
            return self.values
        self.values = np.concatenate([self.values, values])
        return self.values

class Trainer:
    def __init__(self, Model, DataModule, Dataset, Tokenizer, ValidResult, MinLoss, df_train, config, transforms, mlflow_logger):
        # const
        self.mlflow_logger = mlflow_logger
        self.config = config
        self.df_train = df_train
        self.transforms = transforms
        self.skf = StratifiedKFold(
            self.config["n_splits"],
            shuffle=True,
            random_state=self.config["random_seed"])

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer
        self.MinLoss = MinLoss
        self.ValidResult = ValidResult

        # variable
        self.min_loss = self.MinLoss()
        self.val_probs = self.ValidResult()
        self.val_labels = self.ValidResult()

    def run(self):
        for fold, (idx_train, idx_val) in enumerate(self.skf.split(self.df_train, self.df_train[self.config["label"]])):
            # create datamodule
            datamodule = self._create_datamodule(idx_train, idx_val)

            # train crossvalid models
            min_loss = self._train_with_crossvalid(datamodule, fold)
            self.min_loss.update(min_loss)

            # valid
            val_probs, val_labels = self._valid(datamodule, fold)
            self.val_probs.append(val_probs)
            self.val_labels.append(val_labels)

        # log
        self.mlflow_logger.log_metrics({"train_min_loss": self.min_loss.value})

        # train final model
        datamodule = self._create_datamodule_with_alldata()
        self._train_without_valid(datamodule, self.min_loss.value)


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

    def _create_datamodule_with_alldata(self):
        df_val_dummy = self.df_train.iloc[:10]
        datamodule = self.DataModule(
            df_train=self.df_train,
            df_val=df_val_dummy,
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
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
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
            callbacks=[lr_monitor, loss_checkpoint, earystopping],
            **self.config["trainer"],
        )

        trainer.fit(model, datamodule=datamodule)

        self.mlflow_logger.experiment.log_artifact(
            self.mlflow_logger.run_id,
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
            **self.config["trainer"]
        )

        trainer.validate(model, datamodule=datamodule)

        val_probs = model.val_probs
        val_labels = model.val_labels
        return val_probs, val_labels

class Predictor:
    def __init__(self, Model, DataModule, Dataset, Tokenizer, df_test, config, transforms):
        # const
        self.config = config
        self.df_test = df_test
        self.transforms = transforms

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset
        self.Tokenizer = Tokenizer

        # variables
        self.probs = None

    def run(self):
        # create datamodule
        datamodule = self._create_datamodule()

        # predict
        self.probs = self._predict(datamodule)

        return self.probs


    def _create_datamodule(self):
        datamodule = self.DataModule(
            df_train=None,
            df_val=None,
            df_pred=self.df_test,
            Dataset=self.Dataset,
            Tokenizer=self.Tokenizer,
            config=self.config["datamodule"],
            transforms=self.transforms
        )
        return datamodule

    def _predict(self, datamodule):
        # define trainer
        trainer = pl.Trainer(
            logger=None,
            **self.config["trainer"]
        )

        # load model
        model = self.Model.load_from_checkpoint(
            f"{self.config['path']['model_dir']}/{self.config['modelname']}.ckpt",
            config=self.config["model"],
            transforms=self.transforms
        )

        # prediction
        with torch.inference_mode():
            preds = trainer.predict(model, datamodule=datamodule)
        probs = np.concatenate([p["prob"].numpy() for p in preds], axis=0)
        return probs

class PredictorEnsemble(Predictor):
    def _predict(self, datamodule):
        # define trainer
        trainer = pl.Trainer(
            logger=None,
            **self.config["trainer"]
        )

        probs_folds = []
        for fold in range(self.config["n_splits"]):

            # load model
            model = self.Model.load_from_checkpoint(
                f"{self.config['path']['model_dir']}/{self.config['modelname']}_{fold}.ckpt",
                config=self.config["model"],
                transforms=self.transforms
            )

            # prediction
            with torch.inference_mode():
                preds = trainer.predict(model, datamodule=datamodule)
            probs = np.concatenate([p["prob"].numpy() for p in preds], axis=0)
            probs_folds.append(probs)
        probs_ensemble = np.mean(probs_folds, axis=0)
        return probs_ensemble


class ConfusionMatrix:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels

        # variables
        self.fig = plt.figure(figsize=[4, 4], tight_layout=True)

    def draw(self):
        idx_probs = np.argmax(self.probs, axis=1)
        idx_labels = np.argmax(self.labels, axis=1)

        df_confmat = pd.DataFrame(
            confusion_matrix(idx_probs, idx_labels),
            index=self.config["label"],
            columns=self.config["label"]
        )
        axis = self.fig.add_subplot(1, 1, 1)
        sns.heatmap(df_confmat, ax=axis, cmap="bwr", square=True, annot=True)
        axis.set_xlabel("label")
        axis.set_ylabel("pred")
        return self.fig

class F1Score:
    def __init__(self, probs, labels, config):
        # const
        self.config = config
        self.probs = probs
        self.labels = labels

        # variables
        self.f1_scores = {
            "macro": None,
            "micro": None
        }

    def calc(self):
        idx_probs = np.argmax(self.probs, axis=1)
        idx_labels = np.argmax(self.labels, axis=1)
        self.f1_scores = {
            "macro": f1_score(idx_probs, idx_labels, average="macro"),
            "micro": f1_score(idx_probs, idx_labels, average="micro")
        }
        return self.f1_scores

class LogLoss:
    def __init__(self, probs, labels, config):
        # const
        self.probs = probs
        self.labels = labels
        self.config = config
        self.prob_min = 10**(-15)
        self.prob_max = 1-10**(-15)

        # variables
        self.logloss = np.nan

    def calc(self):
        norm_probs = self.probs / np.sum(self.probs, axis=1)[:, None]
        log_probs = np.log(np.clip(norm_probs, self.prob_min, self.prob_max))
        self.logloss = -np.mean(self.labels * log_probs)
        return self.logloss


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

def update_model(config):
    filepaths_ckpt = glob.glob(f"{config['path']['temporal_dir']}/*.ckpt")
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    for filepath_ckpt in filepaths_ckpt:
        filename = pathlib.Path(filepath_ckpt).name
        shutil.move(filepath_ckpt, str(dirpath_model / filename))

if __name__=="__main__":

    # preprocessor
    text_cleaner = TextCleaner()

    if config["mode"]=="train":

        # logger
        mlflow_logger = create_mlflow_logger(config)
        mlflow_logger.log_hyperparams(config)

        # Setting Dataset
        df_train = pd.read_csv(config["path"]["traindata"])
        df_train["essay"] = df_train["essay_id"].apply(fetchEssay, args=(config["path"]["trainessay"],))
        df_train = text_cleaner.clean(df_train, "discourse_text")
        df_train = text_cleaner.clean(df_train, "essay")

        # Training
        trainer = Trainer(
            PeModel,
            PeDataModule,
            PeDataset,
            AutoTokenizer,
            ValidResult,
            MinLoss,
            df_train,
            config,
            None,
            mlflow_logger
        )
        trainer.run()

        # Validation Result
        confmat = ConfusionMatrix(
            trainer.val_probs.values,
            trainer.val_labels.values,
            config["Metrics"]
        )
        fig_confmat = confmat.draw()
        fig_confmat.savefig(f"{config['path']['temporal_dir']}/confmat.png")
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id,
            f"{config['path']['temporal_dir']}/confmat.png"
        )

        f1score = F1Score(
            trainer.val_probs.values,
            trainer.val_labels.values,
            config["Metrics"]
        )
        f1scores = f1score.calc()
        mlflow_logger.log_metrics({
            "macro_f1_score": f1scores["macro"],
            "micro_f1_score": f1scores["micro"]
        })

        logloss = LogLoss(
            trainer.val_probs.values,
            trainer.val_labels.values,
            config["Metrics"]
        )
        log_loss = logloss.calc()
        mlflow_logger.log_metrics({
            "logloss": log_loss
        })

        # output
        print(f"macro_f1_score: {f1scores['macro']:.04f}")
        print(f"micro_f1_score: {f1scores['micro']:.04f}")
        print(f"logloss: {log_loss:.04f}")

        # update model
        update_model(config)


    if config["mode"]=="test":

        # Setting Dataset
        df_test = pd.read_csv(config["path"]["testdata"])
        df_test["essay"] = df_test["essay_id"].apply(fetchEssay, args=(config["path"]["testessay"],))
        df_test = text_cleaner.clean(df_test, "discourse_text")
        df_test = text_cleaner.clean(df_test, "essay")

        # Prediction
        if config["pred_ensemble"]:
            cls_predictor = PredictorEnsemble
        else:
            cls_predictor = Predictor
        predictor = cls_predictor(
            PeModel,
            PeDataModule,
            PeDataset,
            AutoTokenizer,
            df_test,
            config,
            None
        )
        predictor.run()

        # output
        submission = pd.concat([
            df_test["discourse_id"],
            pd.DataFrame(predictor.probs, columns=config["labels"])
        ], axis=1)
        submission.to_csv("submission.csv", index=None)
