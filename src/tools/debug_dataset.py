import os
import random
import sys
import pathlib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from config.distilbert import config
from components.preprocessor import TextCleaner, DataPreprocessor
from components.datamodule import FpDataset

def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

data_preprocessor = DataPreprocessor(TextCleaner, config)
df_train = data_preprocessor.train_dataset()

# FpDataSet
fix_seed(config["random_seed"])
dataset = FpDataset(df_train, config["datamodule"]["dataset"], AutoTokenizer)
batch = dataset.__getitem__(0)


#
filepath_train = "/workspace/kaggle/input/feedback-prize-effectiveness/train.csv"
df_train = pd.read_csv(filepath_train)

filepath_org_and_jp = "/workspace/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_org_and_backtrans_jp.csv"
df_train_org_and_jp = pd.read_csv(filepath_org_and_jp)

filepath_org_and_jp_ojosama = "/workspace/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_org_and_backtrans_jp_ojosama.csv"
df_train_org_and_jp_ojosama = pd.read_csv(filepath_org_and_jp_ojosama)
df_train_org_and_jp_ojosama[df_train_org_and_jp_ojosama["essay_id"].isna()]

filepath_ojosama = "/workspace/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_ojosama.csv"
df_train_ojosama = pd.read_csv(filepath_ojosama)
df_train_ojosama[df_train_ojosama["essay_id"].isna()]
df_train_ojosama_ = df_train_ojosama.dropna()
df_train_ = df_train.copy()
df_train_["discourse_text"] = np.nan
df_train_["essay_id"] = df_train_["essay_id"] + "_ojosama"
df_train_["discourse_text"] = df_train_ojosama_["discourse_text"]
df_train_.to_csv(filepath_ojosama, index=None)

filepath_backtrans_ojosama = "/workspace/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_backtrans_ojosama.csv"
df_train_backtrans_ojosama = pd.read_csv(filepath_backtrans_ojosama)
df_train_backtrans_ojosama_ = df_train_backtrans_ojosama.dropna()
df_train_ = df_train.copy()
df_train_["discourse_text"] = np.nan
df_train_["essay_id"] = df_train_["essay_id"] + "_1"
df_train_["discourse_text"] = df_train_backtrans_ojosama_["discourse_text"]
df_train_[df_train_["discourse_text"].isna()]
df_train_.to_csv(filepath_backtrans_ojosama, index=None)


filepath_jp = "/workspace/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_jp.csv"
df_train_jp = pd.read_csv(filepath_jp)
df_train_jp[df_train_jp["essay_id"].isna()]

filepath_org_and_jp_ojosama = "/workspace/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_org_and_backtrans_jp_ojosama.csv"
df_train_org_and_jp_ojosama = pd.read_csv(filepath_org_and_jp_ojosama)
df_train_org_and_jp_ojosama.loc[df_train_org_and_jp_ojosama["essay_group"].isna(), "essay_group"] = \
df_train_org_and_jp_ojosama.loc[df_train_org_and_jp_ojosama["essay_group"].isna(), "essay_id"].str.replace("_1", "")
df_train_org_and_jp_ojosama["essay_group"].isna().any()
df_train_org_and_jp_ojosama = pd.concat([df_train_org_and_jp, df_train_backtrans_ojosama]).reset_index(drop=True)
df_train_org_and_jp_ojosama[df_train_org_and_jp_ojosama["discourse_text"].isna()]
df_train_org_and_jp_ojosama[df_train_org_and_jp_ojosama["essay_id"].isna()]
df_train_org_and_jp_ojosama.to_csv(filepath_org_and_jp_ojosama, index=None)
