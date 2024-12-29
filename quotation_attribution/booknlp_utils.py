import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
import pickle as pkl

from gen_utils import *

OP_READ_ROOT = 'booknlpen/pdnc_output'

def read_booknlp_op(path):
    return pd.read_csv(path, \
                    sep='\t', quoting=3, lineterminator='\n')

def read_entdf(novel, read_root=None):
    if not read_root:
        read_root = OP_READ_ROOT
    return read_booknlp_op(os.path.join(read_root, novel, novel+'.entities'))

def read_qdf(novel, read_root=None):
    if not read_root:
        read_root = OP_READ_ROOT
    return read_booknlp_op(os.path.join(read_root, novel, novel+'.quotes'))

def read_tokdf(novel, read_root=None):
    if not read_root:
        read_root = OP_READ_ROOT
    return read_booknlp_op(os.path.join(read_root, novel, novel+'.tokens'))



