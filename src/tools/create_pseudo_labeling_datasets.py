import os
import shutil
import sys
import argparse
import glob
import datetime
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import traceback

if __name__=="__main__":

    # datasets for pseudo labeling
    filepath_train = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_org_and_backtrans_jp.csv"
    filepath_pseudo = "/kaggle/input/feedback-prize-2021-unduplicated/train.csv"
    dirpath_output = "/kaggle/input/psuedo-feedback-prize-2021/"
    discourse_effectiveness = ['Adequate', 'Ineffective', 'Effective']

    df_train = pd.read_csv(filepath_train)
    df_pseudo = pd.read_csv(filepath_pseudo)
    df_pseudo["essay_group"] = df_pseudo["essay_id"]

    df_train["pseudo"] = False
    df_pseudo["pseudo"] = True
    df_pseudo["discourse_id"] = df_pseudo["discourse_id"].astype(int)
    df_pseudo["discourse_effectiveness"] = np.random.choice(discourse_effectiveness, len(df_pseudo))
    df_train_and_pseudo = pd.concat([df_train, df_pseudo]).reset_index(drop=True)

    df_train.to_csv(f"{dirpath_output}/train.csv", index=None)
    df_pseudo.to_csv(f"{dirpath_output}/pseudo.csv", index=None)
    df_train_and_pseudo.to_csv(f"{dirpath_output}/train_and_pseudo.csv", index=None)

    # datasets for debug
    dirpath_output = "/kaggle/input/debug-psuedo/"

    df_train_debug = df_train.head(64)
    df_pseudo_debug = df_pseudo.head(64)
    df_train_debug.to_csv(f"{dirpath_output}/train.csv", index=None)
    df_pseudo_debug.to_csv(f"{dirpath_output}/psuedo.csv", index=None)

    df_train_and_pseudo_debug = pd.concat([df_train_debug, df_pseudo_debug]).reset_index(drop=True)
    df_train_and_pseudo_debug.to_csv(f"{dirpath_output}/train_and_pseudo.csv", index=None)

    # link essays
    dirpath_txt = "/kaggle/input/feedback-prize-effectiveness/train/"
    dirpath_txt_2021 = "/kaggle/input/feedback-prize-2021-unduplicated/train/"
    dirpath_txt_output = "/kaggle/input/debug-psuedo/train/"
    for essay_id in tqdm(df_train_debug["essay_id"].unique()):
        filename_essay = f"{essay_id}.txt"
        filepath_essay = f"{dirpath_txt}/{filename_essay}"
        os.symlink(filepath_essay, f"{dirpath_txt_output}/{filename_essay}")
    for essay_id in tqdm(df_pseudo_debug["essay_id"].unique()):
        filename_essay = f"{essay_id}.txt"
        filepath_essay = f"{dirpath_txt_2021}/{filename_essay}"
        os.symlink(filepath_essay, f"{dirpath_txt_output}/{filename_essay}")
