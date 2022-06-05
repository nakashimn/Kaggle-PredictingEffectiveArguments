import os
import sys
import glob
from tqdm import tqdm
import numpy as np
from scipy.special import softmax
import pandas as pd

if __name__=="__main__":

    filepath_test_csv = "/kaggle/input/feedback-prize-effectiveness/test.csv"

    df_test = pd.read_csv(filepath_test_csv)

    list_preds = []

    for idx in df_test.index:
        probs = softmax(np.random.rand(3))
        preds = {
            "discourse_id": df_test.loc[idx].discourse_id,
            "Ineffective": probs[0],
            "Adequate": probs[1],
            "Effective": probs[2]
        }
        list_preds.append(preds)

    df_preds = pd.DataFrame(list_preds)

    df_preds.to_csv("submission.csv", index=None)
