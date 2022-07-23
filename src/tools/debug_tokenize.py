import os
import sys
import shutil
from tqdm import tqdm
import pathlib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import traceback

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.distilbert import config
from components.preprocessor import TextCleaner, DataPreprocessor

# prepare
tokenizer = AutoTokenizer.from_pretrained(
    config["datamodule"]["dataset"]["base_model_name"],
    use_fast=config["datamodule"]["dataset"]["use_fast_tokenizer"]
)
df_train = pd.read_csv(config["path"]["traindata"])

data_preprocessor = DataPreprocessor(TextCleaner, config)
df_train = data_preprocessor.train_dataset()

# tokenize
for idx in tqdm(df_train.index):
    text = df_train.loc[idx, "discourse_type"]
    try:
        token = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=config["datamodule"]["dataset"]["max_length"],
            padding="max_length"
        )
        ids = torch.tensor([token["input_ids"]])
        masks = torch.tensor([token["attention_mask"]])
    except:
        print(text)
        print(traceback.format_exc())

pbar = tqdm(df_train.index)
for idx in pbar:
    essay_id = df_train.loc[idx, "essay_id"]
    pbar.set_postfix_str(essay_id)

    filepath_essay = str(pathlib.Path(config["path"]["trainessay"])/ f"{essay_id}.txt")
    if not os.path.exists(filepath_essay):
        filepath_essay_org = filepath_essay.replace("_0", "")
        shutil.copy2(filepath_essay_org, filepath_essay)
        print(f"copy {filepath_essay_org}")
    with open(filepath_essay, "r") as f:
        text = f.read()
    try:
        token = tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=config["datamodule"]["dataset"]["max_length"],
            padding="max_length"
        )
        ids = torch.tensor([token["input_ids"]])
        masks = torch.tensor([token["attention_mask"]])
    except:
        print(text)
        print(traceback.format_exc())
        break
