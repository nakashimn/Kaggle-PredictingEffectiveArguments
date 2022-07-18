import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import traceback

class FpDatasetTokenized(Dataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.tokenizer = Tokenizer.from_pretrained(
            self.config["base_model_name"],
            use_fast=self.config["use_fast_tokenizer"]
        )
        self.ids, self.masks = self.read_values(df)
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self.read_labels(df)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ids = torch.tensor(self.ids[idx].astype(np.int32), dtype=torch.long)
        masks = torch.tensor(self.masks[idx], dtype=torch.long)
        if self.transform is not None:
            ids = self.transform(ids)
        if self.labels is not None:
            labels = self.labels[idx]
            return ids, masks, labels
        return ids, masks

    def read_values(self, df):
        texts = (
            df["discourse_type"]+ " " + df["discourse_text"] + " " + df["essay"]
        ).values
        ids = []
        masks = []
        for text in texts:
            token = self.tokenize(text)
            ids.append(token["input_ids"])
            masks.append(token["attention_mask"])
        ids = np.array(ids, dtype=np.uint16)
        masks = np.array(masks, dtype=np.int8)
        return ids, masks

    def read_labels(self, df):
        labels = F.one_hot(
            torch.tensor(
                [self.config[self.config["label"]][d] for d in df[self.config["label"]]]
            ),
            num_classes=self.config["num_class"]
        ).float()
        return labels

    def tokenize(self, text):
        token = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding="max_length"
        )
        return token

class FpDataset(Dataset):
    def __init__(self, df, config, Tokenizer, transform=None):
        self.config = config
        self.val = self.read_values(df)
        self.labels = None
        if self.config["label"] in df.keys():
            self.labels = self.read_labels(df)
        self.tokenizer = Tokenizer.from_pretrained(
            self.config["base_model_name"],
            use_fast=self.config["use_fast_tokenizer"]
        )
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

    def read_values(self, df):
        values = (
            df["discourse_type"]+ " " + df["discourse_text"] + " " + df["essay"]
        ).values
        return values

    def read_labels(self, df):
        labels = F.one_hot(
            torch.tensor(
                [self.config[self.config["label"]][d] for d in df[self.config["label"]]]
            ),
            num_classes=self.config["num_class"]
        ).float()
        return labels

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

class FpDataModule(LightningDataModule):
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
        dataset = self.Dataset(
            self.df_train,
            self.config["dataset"],
            self.Tokenizer,
            self.transforms["train"]
        )
        return DataLoader(dataset, **self.config["train_loader"])

    def val_dataloader(self):
        dataset = self.Dataset(
            self.df_val,
            self.config["dataset"],
            self.Tokenizer,
            self.transforms["valid"]
        )
        return DataLoader(dataset, **self.config["val_loader"])

    def predict_dataloader(self):
        dataset = self.Dataset(
            self.df_pred,
            self.config["dataset"],
            self.Tokenizer,
            self.transforms["pred"]
        )
        return DataLoader(dataset, **self.config["pred_loader"])
