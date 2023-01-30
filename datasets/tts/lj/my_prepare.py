# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

os.environ["OMP_NUM_THREADS"] = "1"

import json
import os
import re
import subprocess
from multiprocessing.pool import Pool
import pandas as pd
from g2p_en import G2p
from tqdm import tqdm

basedir = '/u01/stt/tts/small-f2s/2022_norm/2022/norm/'
# basedir = "/u01/stt/tts/small-f2s/PriorGrad-acoustic/LJSpeech-1.1"
lexicon_path = "/u01/stt/tts/small-f2s/2022_norm/2022/norm/lexicon.txt"
# g2p = G2p()

def get_ph(lexicon_path):
    with open(lexicon_path, "r+") as f:
        list_ph = f.read().splitlines()

    dict_rs = {}
    for line in list_ph:
        ph_split = line.split("\t")
        dict_rs[ph_split[0]] = ph_split[1]
    return dict_rs
dict_ph = get_ph(lexicon_path)

def v2p(text):
    text = re.sub("\s+"," ", text)
    text = re.sub("([\!\'\,\-\.\?])+", " . ", text)
    rs = []
    list_word = dict_ph.keys()
    for word in text.strip().split(" "):
        if word in list_word:
            rs.append(dict_ph[word])
    return " | ".join(rs)

print(v2p("sơn đẹp cứt"))