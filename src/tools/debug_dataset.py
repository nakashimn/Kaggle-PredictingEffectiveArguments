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
from components.datamodule import FpDataset, FpDatasetTokenized

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

# FpDataSetTokenized
fix_seed(config["random_seed"])
dataset_tokenized = FpDatasetTokenized(df_train, config["datamodule"]["dataset"], AutoTokenizer)
batch_ = dataset_tokenized.__getitem__(0)
