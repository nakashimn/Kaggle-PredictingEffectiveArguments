import os
import sys
import argparse
import glob
import datetime
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

dirpath_train_txt = "/kaggle/input/train/"
filepath_train_csv = "/kaggle/input/train.csv"

df_train = pd.read_csv(filepath_train_csv)


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
