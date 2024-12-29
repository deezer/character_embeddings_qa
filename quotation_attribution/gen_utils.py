import os, re, sys, json, csv, string, gzip
import importlib
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
import pickle as pkl

def remove_untitled(df):
    cols = list(df.columns)
    for col in cols:
        if 'Untitled' in col:
            del df[col]

    return df