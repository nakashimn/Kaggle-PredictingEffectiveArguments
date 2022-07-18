import os
import sys
import glob
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from googletrans import Translator
import matplotlib.pyplot as plt
import traceback

if __name__=="__main__":

    # param
    label = "discourse_text"
    wait_sec = 0.8
    intermediate_language = "ja"

    # path
    filepath_csv = "../kaggle/input/feedback-prize-effectiveness/train.csv"
    filepath_backtrans = "../kaggle/input/back-translated-feedback-prize-effectiveness/train.csv"
    ## 元textの変換は未実施
    # dirpath_txt = "../kaggle/input/feedback-prize-effectiveness/train/"
    # filepaths_txt = glob.glob(f"{dirpath_txt}/*.txt")
    # dirpath_backtrans = "../kaggle/input/back-translated-feedback-prize-effectiveness/train/"

    # read
    df_train = pd.read_csv(filepath_csv)
    df_train_backtrans = pd.read_csv(filepath_backtrans)

    # prepare
    translator = Translator()
    ## Ineffective -> Effective -> Adequate の順
    indices_trans = df_train.sort_values("discourse_effectiveness", ascending=False).index

    # translate
    ## googletransの制限に途中で引っかかる可能性あり
    ## try/except節でError発生時に中間状態を保存して終了
    try:
        for idx in tqdm(indices_trans):
            if isinstance(df_train_backtrans.loc[idx, label], str):
                continue
            text_org = df_train.loc[idx, label]
            try:
                translated = translator.translate(text_org, dest=intermediate_language, src="en")
                time.sleep(wait_sec)
                back_translated = translator.translate(translated.text, dest="en", src=intermediate_language)
                df_train_backtrans.loc[idx, label] = back_translated.text
                time.sleep(wait_sec)
            ## 文字列にNoneが含まれてしまう場合のみスキップして対処
            except TypeError:
                print(traceback.format_exc())
    except:
        print(traceback.format_exc())
    finally:
        df_train_backtrans.to_csv(filepath_backtrans, index=None)
