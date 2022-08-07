import os
import sys
import argparse
import pathlib
import glob
import datetime
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

if __name__=="__main__":

    filepath_train = "/workspace/kaggle/input/feedback-prize-effectiveness/train.csv"
    filepath_train_prev = "/workspace/kaggle/input/feedback-prize-2021/train.csv"

    df_train = pd.read_csv(filepath_train)
    df_train_prev = pd.read_csv(filepath_train_prev)

    essay_id_train = df_train["essay_id"].unique()

    df_train_prev["duplicated"] = df_train_prev["id"].isin(essay_id_train)

    df_train_unduplicated = df_train_prev[~df_train_prev["duplicated"]].reset_index(drop=True)
    df_train_unduplicated = df_train_unduplicated[["discourse_id", "id", "discourse_text", "discourse_type"]]
    df_train_unduplicated = df_train_unduplicated.rename(columns={"id": "essay_id"})

    df_train_unduplicated.to_csv("/workspace/kaggle/input/feedback-prize-2021-unduplicated/train.csv", index=None)

    dirpath_essay_prev = "/workspace/kaggle/input/feedback-prize-2021/train/"
    filepaths_essay_prev = glob.glob(f"{dirpath_essay_prev}/*.txt")
    dirpath_essay_prev_unduplicated = "/workspace/kaggle/input/feedback-prize-2021-unduplicated/train/"

    essay_id_unduplicated = df_train_unduplicated["essay_id"].unique()
    pbar = tqdm(filepaths_essay_prev)
    for fp in pbar:
        file_name = pathlib.Path(fp).name
        file_id = pathlib.Path(fp).stem
        if file_id not in essay_id_unduplicated:
            pbar.set_description_str(f"skip: {file_id}")
            continue
        os.symlink(fp, f"{dirpath_essay_prev_unduplicated}/{file_name}")
