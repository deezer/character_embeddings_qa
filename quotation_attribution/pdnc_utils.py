import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
import pickle as pkl
sys.path.append('/h/vkpriya/qa_eval')
from gen_utils import *

metadf = pd.read_csv('data/pdnc_source/ListOfNovels.txt')
novels = sorted(metadf['folder'].tolist())

DATA_READ_ROOT = 'data/pdnc_source'

IGNORE = ['_unknowable', '_group', '_narr', 'Unknowable', '(Unknown)', 'Unknown']
PREFIXES = ['Mr.', 'Mrs.', 'Miss.', 'Lady', 'Sir', 'Mrs', 'Mr', 'Miss', 'Dr.', 'Dr', 'Madame', 'Madam', \
           'Mademoiselle', 'St.', 'St', 'Ms.', 'Ms', 'Count', 'Countess']
PREFIXES.extend([x.lower() for x in PREFIXES])


def get_enhanced_char_list(name2id, add_lowercase=False):
    #don't include lowercase because names can also be common nouns sometimes (Lily, Rose)
    enhanced_name2id = {}
    
    new_cands = {}
    
    for name, id_ in name2id.items():
        enhanced_name2id[name] = id_
        if add_lowercase:
            enhanced_name2id[name.lower()] = id_
        
    for name, id_ in name2id.items():
        
        n_words = name.split()
        if n_words[0] in PREFIXES:
        # n_words = [x for x in n_words if x not in PREFIXES]
            new_cand = " ".join(n_words[1:])

            if (len(new_cand)>0) and (new_cand not in enhanced_name2id):
                if new_cand not in new_cands:
                    new_cands[new_cand] = []
                new_cands[new_cand].append(id_)

                if add_lowercase:
                    if new_cand.lower() not in new_cands:
                        new_cands[new_cand.lower()] = []

                    new_cands[new_cand.lower()].append(id_)
        
    
    for new_cand, ids in new_cands.items():
        ids = list(set(ids))
        if len(ids) == 1:
            enhanced_name2id[new_cand] = ids[0]
#         else:
#             print("Mult matches: {}, {}".format(new_cand, ids))
    
    print("original count: {} ; enhanced count: {}".format(len(name2id), len(enhanced_name2id)))
    
    enhanced_id2names = {}
    for n, i in enhanced_name2id.items():
        if i not in enhanced_id2names:
            enhanced_id2names[i] = set()
        enhanced_id2names[i].add(n)

    return {'name2id': enhanced_name2id, 'id2names': enhanced_id2names}

def read_qadf(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    return remove_untitled(pd.read_csv(os.path.join(read_root, novel, 'quote_info.csv')))

def read_chapdf(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    return remove_untitled(pd.read_csv(os.path.join(read_root, novel, 'chap_info.csv')))

def read_pardf(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    return remove_untitled(pd.read_csv(os.path.join(read_root, novel, 'para_info.csv')))

def read_madf(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    return remove_untitled(pd.read_csv(os.path.join(read_root, novel, 'mention_info.csv')))

def read_mudf(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    return remove_untitled(pd.read_csv(os.path.join(read_root, novel, 'mentions_used.csv')))

def read_ntext(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    with open(os.path.join(read_root, novel, 'novel.txt'), 'r') as f:
        ntext = f.read().strip()
    return ntext

def read_char_info(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    cinfo = pkl.load(open(os.path.join(read_root, novel, 'charInfo.dict.pkl'), 'rb'))
    return cinfo

def read_enchar_info(novel, read_root=None):
    if not read_root:
        read_root = DATA_READ_ROOT
    try:
        cinfo = pkl.load(open(os.path.join(read_root, novel, 'enhanced_charInfo.dict.pkl'), 'rb'))
        return cinfo
    except FileNotFoundError:
        print("Not found! Creating..")
        cinfo = read_char_info(novel)
        return get_enhanced_char_list(cinfo['name2id'])