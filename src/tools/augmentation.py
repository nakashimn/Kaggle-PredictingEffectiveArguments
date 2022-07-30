import os
import sys
import re
import codecs
import subprocess
import pathlib
import glob
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from googletrans import Translator
import traceback
from text_unidecode import unidecode
import traceback

def replace_encoding_with_utf8(error: UnicodeError):
    return error.object[error.start : error.end].encode("utf-8"), error.end

def replace_decoding_with_cp1252(error: UnicodeError):
    return error.object[error.start : error.end].decode("cp1252"), error.end

codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

def resolve_encodings_and_normalize(text):
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text

def read_text(filepath):
    with open(filepath, "r") as f:
        text = f.read()
    return text

def write_text(filepath, text):
    with open(filepath, "w") as f:
        f.write(text)

def clean_text(text: str):
    text = re.sub("\n+", "\n", text)
    text = re.sub("\t+", " ", text)
    text = text.replace("\xa0", " ")
    text = text.strip()
    return text

def split_text_to_sentences(text, delimiter=". "):
    sentences = text.split(delimiter)
    sentences = [sentence + delimiter for sentence in sentences[:-1]] + [sentences[-1]]
    return sentences

def backtrans_discourse_text(
    filepath_csv,
    filepath_backtrans,
    filepath_trans_jp,
    wait_sec=1,
    intermediate_language="ja"
):
    header_text = "discourse_text"

    # read
    df_train = pd.read_csv(filepath_csv)
    df_train_backtrans = pd.read_csv(filepath_backtrans)
    df_train_trans_jp = pd.read_csv(filepath_trans_jp)

    # prepare
    translator = Translator(raise_exception=True)
    ## Ineffective -> Effective -> Adequate の順
    indices_trans = df_train.sort_values(
        "discourse_effectiveness", ascending=False
    ).index

    # translate
    ## googletransの制限に途中で引っかかる可能性あり
    ## try/except節でError発生時に中間状態を保存して終了
    try:
        for idx in tqdm(indices_trans):
            if isinstance(df_train_trans_jp.loc[idx, header_text], str):
                continue
            text_org = df_train.loc[idx, header_text]
            text_org = resolve_encodings_and_normalize(text_org)
            text_org = clean_text(text_org)
            try:
                translated = translator.translate(
                    text_org,
                    dest=intermediate_language,
                    src="en"
                )
                time.sleep(wait_sec)
                back_translated = translator.translate(
                    translated.text,
                    dest="en",
                    src=intermediate_language
                )
                time.sleep(wait_sec)

                df_train_backtrans.loc[idx, header_text] = back_translated.text
                df_train_trans_jp.loc[idx, header_text] = translated.text

            ## 文字列にNoneが含まれてしまう場合のみスキップして対処
            except TypeError:
                print(traceback.format_exc())
    except:
        print(traceback.format_exc())
    finally:
        df_train_backtrans.to_csv(filepath_backtrans, index=None)
        df_train_trans_jp.to_csv(filepath_trans_jp, index=None)

def backtrans_discourse_text_ojosama(
    filepath_trans_jp,
    filepath_backtrans,
    filepath_trans_ojosama,
    filepath_ojosama_translator,
    wait_sec=1
):
    header_text = "discourse_text"

    # read
    df_trans_jp = pd.read_csv(filepath_trans_jp)
    df_train_backtrans = pd.read_csv(filepath_backtrans)
    df_trans_ojosama = pd.read_csv(filepath_trans_ojosama)

    # prepare
    translator = Translator(raise_exception=True)
    ## Ineffective -> Effective -> Adequate の順
    indices_trans = df_trans_jp.sort_values(
        "discourse_effectiveness", ascending=False
    ).index

    # translate
    ## googletransの制限に途中で引っかかる可能性あり
    ## try/except節でError発生時に中間状態を保存して終了
    try:
        for idx in tqdm(indices_trans):
            if isinstance(df_train_backtrans.loc[idx, header_text], str):
                continue
            text_jp = df_trans_jp.loc[idx, header_text]
            text_jp = clean_text(text_jp)
            try:
                proc = subprocess.run(
                    [filepath_ojosama_translator, "-t", text_jp],
                    stdout=subprocess.PIPE
                )
                text_ojosama = proc.stdout.decode("utf8")
                text_ojosama = clean_text(text_ojosama)
                back_translated = translator.translate(
                    text_ojosama,
                    dest="en",
                    src="ja"
                )
                time.sleep(wait_sec)

                df_train_backtrans.loc[idx, header_text] = back_translated.text
                df_trans_ojosama.loc[idx, header_text] = text_ojosama

            ## 文字列にNoneが含まれてしまう場合のみスキップして対処
            except TypeError:
                print(traceback.format_exc())
    except:
        print(traceback.format_exc())
    finally:
        df_train_backtrans.to_csv(filepath_backtrans, index=None)
        df_trans_ojosama.to_csv(filepath_trans_ojosama, index=None)


def backtrans_essay(
    dirpath_train,
    dirpath_backtrans,
    dirpath_trans_jp,
    wait_sec=1,
    intermediate_language="ja"
):
    # read
    filepaths_txt = glob.glob(f"{dirpath_train}/*.txt")

    # prepare
    translator = Translator(raise_exception=True)

    ## googletransの制限に途中で引っかかる可能性あり
    pbar = tqdm(filepaths_txt)
    for filepath_txt in pbar:
        file_id = pathlib.Path(filepath_txt).stem
        pbar.set_postfix_str(file_id)

        filepath_backtrans = str(pathlib.Path(dirpath_backtrans) / f"{file_id}_0.txt")
        filepath_trans_jp = str(pathlib.Path(dirpath_trans_jp) / f"{file_id}_jp.txt")

        if os.path.exists(filepath_backtrans):
            continue

        text_org = read_text(filepath_txt)
        text_org = resolve_encodings_and_normalize(text_org)
        text_org = clean_text(text_org)

        try:
            translated = translator.translate(
                text_org,
                dest=intermediate_language,
                src="en"
            )
            time.sleep(wait_sec)
            back_translated = translator.translate(
                translated.text,
                dest="en",
                src=intermediate_language
            )
            time.sleep(wait_sec)

            write_text(filepath_backtrans, back_translated.text)
            write_text(filepath_trans_jp, translated.text)

        ## 文字列にNoneが含まれてしまう場合のみスキップして対処
        except TypeError:
            print(traceback.format_exc())

def backtrans_long_essay(
    dirpath_train,
    dirpath_backtrans,
    dirpath_trans_jp,
    wait_sec=1,
    intermediate_language="ja"
):
    # read
    filepaths_txt = glob.glob(f"{dirpath_train}/*.txt")

    # prepare
    translator = Translator(raise_exception=True)

    ## googletransの制限に途中で引っかかる可能性あり
    pbar = tqdm(filepaths_txt)
    for filepath_txt in pbar:
        file_id = pathlib.Path(filepath_txt).stem
        pbar.set_postfix_str(file_id)

        filepath_backtrans = str(pathlib.Path(dirpath_backtrans) / f"{file_id}_0.txt")
        filepath_trans_jp = str(pathlib.Path(dirpath_trans_jp) / f"{file_id}_jp.txt")

        if os.path.exists(filepath_backtrans):
            continue

        text_org = read_text(filepath_txt)
        text_org = resolve_encodings_and_normalize(text_org)
        text_org = clean_text(text_org)

        sentences = split_text_to_sentences(text_org)
        split_point = int(len(sentences) / 2)
        texts_org = [
            "".join(sentences[:split_point]),
            "".join(sentences[split_point:])
        ]

        try:
            texts_backtrans = []
            texts_jp = []
            for text_org in texts_org:
                translated = translator.translate(
                    text_org,
                    dest=intermediate_language,
                    src="en"
                )
                time.sleep(wait_sec)
                back_translated = translator.translate(
                    translated.text,
                    dest="en",
                    src=intermediate_language
                )
                time.sleep(wait_sec)
                texts_backtrans.append(back_translated.text)
                texts_jp.append(translated.text)
            text_backtrans = "".join(texts_backtrans)
            text_jp = "".join(texts_jp)

            write_text(filepath_backtrans, text_backtrans)
            write_text(filepath_trans_jp, text_jp)

        ## 文字列にNoneが含まれてしまう場合のみスキップして対処
        except TypeError:
            print(traceback.format_exc())


def trans_jp_to_ojosama(
    dirpath_trans_jp,
    dirpath_backtrans,
    dirpath_trans_ojosama,
    filepath_ojosama_translator,
    wait_sec=1
):
    # path
    filepaths_txt = glob.glob(f"{dirpath_trans_jp}/*.txt")

    # prepare
    translator = Translator(raise_exception=True)

    # translate
    for fp in tqdm(filepaths_txt):
        filename_ojosama = pathlib.Path(fp).name.replace("_jp", "_ojosama")
        filename_backtrans = pathlib.Path(fp).name.replace("_jp", "_1")
        filepath_ojosama = f"{dirpath_trans_ojosama}/{filename_ojosama}"
        filepath_backtrans = f"{dirpath_backtrans}/{filename_backtrans}"

        if os.path.exists(filepath_backtrans):
            continue

        text_jp = read_text(fp)
        text_jp = clean_text(text_jp)

        try:
            # translate jp to ojousama
            proc = subprocess.run(
                [filepath_ojosama_translator, "-t", text_jp],
                stdout=subprocess.PIPE
            )
            text_ojosama = proc.stdout.decode("utf8")
            text_ojosama = clean_text(text_ojosama)

            # write
            translated = translator.translate(text_ojosama, dest="en", src="ja")
            time.sleep(wait_sec)

            write_text(filepath_ojosama, text_ojosama)
            write_text(filepath_backtrans, translated.text)

        except TypeError:
            print(traceback.format_exc())

def trans_long_jp_to_ojosama(
    dirpath_trans_jp,
    dirpath_backtrans,
    dirpath_trans_ojosama,
    filepath_ojosama_translator,
    wait_sec=1
):
    # path
    filepaths_txt = glob.glob(f"{dirpath_trans_jp}/*.txt")

    # prepare
    translator = Translator(raise_exception=True)

    # translate
    for fp in tqdm(filepaths_txt):
        filename_ojosama = pathlib.Path(fp).name.replace("_jp", "_ojosama")
        filename_backtrans = pathlib.Path(fp).name.replace("_jp", "_1")
        filepath_ojosama = f"{dirpath_trans_ojosama}/{filename_ojosama}"
        filepath_backtrans = f"{dirpath_backtrans}/{filename_backtrans}"

        if os.path.exists(filepath_backtrans):
            continue

        text_jp = read_text(fp)
        text_jp = clean_text(text_jp)

        sentences = split_text_to_sentences(text_ojosama, delimiter="。")
        split_point = int(len(sentences) / 2)
        texts_jp = [
            "".join(sentences[:split_point]),
            "".join(sentences[split_point:])
        ]

        try:
            texts_backtrans = []
            texts_ojosama = []
            for text_jp in texts_jp:
                # translate jp to ojousama
                proc = subprocess.run(
                    [filepath_ojosama_translator, "-t", text_jp],
                    stdout=subprocess.PIPE
                )
                txt_oj = proc.stdout.decode("utf8")
                txt_oj = clean_text(txt_oj)

                # write
                translated = translator.translate(txt_oj, dest="en", src="ja")
                time.sleep(wait_sec)

                texts_backtrans.append(translated.text)
                texts_ojosama.append(txt_oj)

            text_backtrans = "".join(texts_backtrans)
            text_ojosama = "".join(texts_ojosama)

            write_text(filepath_ojosama, text_ojosama)
            write_text(filepath_backtrans, text_backtrans)

        except TypeError:
            print(traceback.format_exc())


if __name__=="__main__":

    # param
    wait_sec = 0.3
    intermediate_language = "ja"

    # path
    filepath_csv = "/kaggle/input/feedback-prize-effectiveness/train.csv"
    filepath_backtrans_jp = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_backtrans_jp.csv"
    filepath_trans_jp = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_jp.csv"
    filepath_backtrans_ojosama = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_backtrans_ojosama.csv"
    filepath_trans_ojosama = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_ojosama.csv"
    dirpath_train = "/kaggle/input/feedback-prize-effectiveness/train/"
    dirpath_backtrans = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train/"
    dirpath_trans_jp = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_jp/"
    dirpath_trans_ojosama = "/kaggle/input/back-translated-feedback-prize-effectiveness-v4/train_ojosama/"
    filepath_ojosama_translator = "/workspace/tools/ojosama"

    # back translate discourse_text
    print("--- backtrans discourse_text / en -> jp -> en ---")
    backtrans_discourse_text(
        filepath_csv,
        filepath_backtrans_jp,
        filepath_trans_jp,
        wait_sec,
        intermediate_language
    )

    # back translate discourse_text ojosama
    print("--- backtrans discourse_text / jp -> ojosama -> en ---")
    backtrans_discourse_text_ojosama(
        filepath_trans_jp,
        filepath_backtrans_ojosama,
        filepath_trans_ojosama,
        filepath_ojosama_translator,
        wait_sec
    )

    # back translate essay
    print("--- backtrans essay / en -> jp -> en ---")
    backtrans_essay(
        dirpath_train,
        dirpath_backtrans,
        dirpath_trans_jp,
        wait_sec,
        intermediate_language
    )

    backtrans_long_essay(
        dirpath_train,
        dirpath_backtrans,
        dirpath_trans_jp,
        wait_sec,
        intermediate_language
    )

    print("--- backtrans essay / jp -> ojosama -> en ---")
    trans_jp_to_ojosama(
        dirpath_trans_jp,
        dirpath_backtrans,
        dirpath_trans_ojosama,
        filepath_ojosama_translator,
        wait_sec
    )

    trans_long_jp_to_ojosama(
        dirpath_trans_jp,
        dirpath_backtrans,
        dirpath_trans_ojosama,
        filepath_ojosama_translator,
        wait_sec
    )
