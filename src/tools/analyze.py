import os
import sys
import pathlib
import argparse
import glob
import datetime
import json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

dirpath_train_txt = pathlib.Path("/kaggle/input/feedback-prize-effectiveness/train/")
filepath_train_csv = "/kaggle/input/feedback-prize-effectiveness/train.csv"

df_train = pd.read_csv(filepath_train_csv)

tokenizer = AutoTokenizer.from_pretrained("/kaggle/input/roberta-base")
masks = []
for idx in df_train.index:
    tokens = tokenizer.encode_plus(
        df_train.loc[idx, "discourse_text"],
        truncation=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length"
    )
    masks.append(tokens["attention_mask"])


tokens = tokenizer.encode_plus(
    df_train.loc[0, "discourse_text"],
    truncation=True,
    add_special_tokens=True,
    max_length=512,
    padding="max_length"
)
ids = tokens["input_ids"]

counts_tokens = np.array(masks).sum(axis=1)
np.quantile(counts_tokens, 0.99)

def read_text(essay_id):
    with open(dirpath_train_txt / f"{essay_id}.txt") as f:
        text = f.read()
    return text

essay_masks = []
for idx in df_train.index:
    tokens = tokenizer.encode_plus(
        read_text(df_train.loc[idx, "essay_id"]),
        truncation=True,
        add_special_tokens=True,
        max_length=512,
        padding="max_length"
    )
    essay_masks.append(tokens["attention_mask"])

counts_essay_tokens = np.array(essay_masks).sum(axis=1)
plt.hist(counts_tokens, bins=30, alpha=0.3)
plt.hist(counts_essay_tokens, bins=30, alpha=0.5)

df_train["discourse_type"] + " " + df_train["discourse_text"]

# type distribution
counts_type = df_train["discourse_type"].value_counts()

# label distribution
counts_label = df_train["discourse_effectiveness"].value_counts()

# type-label crosstab
crosstab_type_label = pd.crosstab(df_train["discourse_type"], df_train["discourse_effectiveness"])
plt.figure(figsize=[4, 4])
sns.heatmap(crosstab_type_label, cmap="bwr", square=True, annot=True,fmt="0")

#
df_train[df_train["essay_id"]=="0A6C0B6D3925"]
