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
    text = re.sub("([\!\'\-\.\?\:\;])+", " . ", text)
    text = text.replace(",", " , ")
    text = re.sub(" +", " ", text)
    rs = []
    list_word = dict_ph.keys()
    for word in text.strip().split(" "):
        if word in list_word:
            rs.append(dict_ph[word])
        else:
            rs.append(word)
    return " | ".join([x for x in rs if (x != "" and x != " ")])

def g2p_job(idx, fn, txt):
    spk = idx // 100

    phs_str = v2p(txt)
    os.makedirs(f'{basedir}/mfa_input/{spk}', exist_ok=True)
    with open(f'{basedir}/mfa_input/{spk}/{fn}.ph', 'w') as f_txt:
        phs_str = re.sub("([\!\'\,\-\.\?])+", r"\1", phs_str)
        f_txt.write(phs_str)
    with open(f'{basedir}/mfa_input/{spk}/{fn}.lab', 'w') as f_txt:
        phs_str_mfa = re.sub("[\!\'\,\-\.\?]+", "PUNC", phs_str)
        phs_str_mfa = re.sub("\|", "SEP", phs_str_mfa)
        phs_str_mfa = re.sub(" +", " ", phs_str_mfa)
        f_txt.write(phs_str_mfa)
    subprocess.check_call(f'cp "{basedir}/wavs/{fn}.wav" "{basedir}/mfa_input/{spk}/"', shell=True)
    return phs_str.split(" "), phs_str_mfa.split(" ")


if __name__ == "__main__":
    # build mfa_input for forced alignment
    p = Pool(os.cpu_count())
    subprocess.check_call(f'rm -rf {basedir}/mfa_input', shell=True)
    futures = []

    # f = open('ljspeech_text.txt', 'w')
    for idx, l in enumerate(open(f'{basedir}/metadata.csv').readlines()):
        fn, _, txt = l.strip().split("|")
        futures.append(p.apply_async(g2p_job, args=[idx, fn, txt]))
    p.close()
    mfa_dict = set()
    phone_set = set()
    # print(futures)
    for f in tqdm(futures):
        phs, phs_mfa = f.get()
        for ph in phs:
            phone_set.add(ph)
        for ph_mfa in phs_mfa:
            mfa_dict.add(ph_mfa)
    mfa_dict = sorted(mfa_dict)
    phone_set = sorted(phone_set)
    print("| mfa dict: ", mfa_dict)
    print("| phone set: ", phone_set)
    with open(f'{basedir}/dict_mfa.txt', 'w') as f:
        for ph in mfa_dict:
            f.write(f'{ph} {ph}\n')
    with open(f'{basedir}/dict.txt', 'w') as f:
        for ph in phone_set:
            f.write(f'{ph} {ph}\n')
    phone_set = ["<pad>", "<EOS>", "<UNK>"] + phone_set
    json.dump(phone_set, open(f'{basedir}/phone_set.json', 'w'))
    p.join()

    # build metadata_phone
    meta_ori_df = pd.read_csv(os.path.join(basedir, 'metadata.csv'), delimiter='|', names=['wav', 'txt1', 'txt2'])
    subprocess.check_call(f"mkdir -p {basedir}/mfa_input_txt; "
                          f"cp {basedir}/mfa_input/*/*.ph {basedir}/mfa_input_txt",
                          shell=True)

    meta_ori_df['phone2'] = meta_ori_df.apply(
        lambda r: open(f"{basedir}/mfa_input_txt/{r['wav']}.ph").readlines()[0].strip(), 1)
    meta_ori_df.to_csv(f"{basedir}/metadata_phone.csv")
