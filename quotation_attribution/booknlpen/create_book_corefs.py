import os, re, sys, json, csv, string, gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

import pickle as pkl

DATA_ROOT = '/data/datasets/project-dialogism-novel-corpus/aligned_data/'

COREF_ROOT = 'pdnc_output'
SAVE_ROOT = COREF_ROOT

IGNORE = ['_unknowable', '_group', '_narr', 'Unknowable', '(Unknown)', 'Unknown']
PREFIXES = ['Mr.', 'Mrs.', 'Miss.', 'Lady', 'Sir', 'Mrs', 'Mr', 'Miss', 'Dr.', 'Dr', 'Madame', 'Madam', \
           'Mademoiselle', 'St.', 'St', 'Ms.', 'Ms', 'Count', 'Countess']
PREFIXES.extend([x.lower() for x in PREFIXES])

novels = []
for nf in os.scandir(DATA_ROOT):
    if os.path.isdir(nf) and nf.name[0] not in ['.', '_']:
        novels.append(nf.name)
novels = sorted(novels)

def read_quote_df(novel):
    df = pd.read_csv(os.path.join(DATA_ROOT, novel, 'quote_info.csv'))
    return df


def read_char_info(novel):
    charInfo = pkl.load(open(os.path.join(DATA_ROOT, novel, 'charInfo.dict.pkl'), 'rb'))
    return charInfo


def read_booknlp_df(path):
    df = pd.read_csv(path, delimiter='\t', quoting=3, lineterminator='\n')
    return df


def get_ntext(novel):
    with open(os.path.join(DATA_ROOT, novel, 'novel.txt'), 'r') as f:
        ntext = f.read().strip()
    return ntext

def get_enhanced_char_list(name2id):
    
    enhanced_name2id = {}
    
    new_cands = {}
    
    for name, id_ in name2id.items():
        enhanced_name2id[name] = id_
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
    return enhanced_name2id


def booknlp_process_novel(novel):
    print(novel)

    if not os.path.isdir(os.path.join(SAVE_ROOT, novel)):
        os.mkdir(os.path.join(SAVE_ROOT, novel))

    entdf_path = os.path.join(COREF_ROOT, novel, novel+'.entities')
    entdf = read_booknlp_df(entdf_path)

    entdf.set_index('COREF', inplace=True)

    tokdf_path = os.path.join(COREF_ROOT, novel, novel+'.tokens')
    tokdf = read_booknlp_df(tokdf_path)

    tokdf.set_index('token_ID_within_document', inplace=True)


    entdf['start_byte'] = [tokdf.loc[x]['byte_onset'] for x in entdf['start_token']]
    entdf['end_byte'] = [tokdf.loc[x]['byte_offset'] for x in entdf['end_token']]

    ntext = get_ntext(novel)

    coref2texts = {}
    for ent in entdf.index.unique():
        edf = entdf.loc[ent]
        if isinstance(edf, pd.core.series.Series):
            coref2texts[ent] = [edf['text']]
        else:
            coref2texts[ent] = edf['text'].tolist()

    charInfo = read_char_info(novel)
    e_name2id = get_enhanced_char_list(charInfo['name2id'])
    charNames = set(e_name2id.keys())

    count_mult = 0
    count_none = 0
    count_match = 0
    rows = []

    for coref, texts in coref2texts.items():
        matches = charNames.intersection(set(texts))
        match_ids = [e_name2id[x] for x in matches]
        if len(set(match_ids))>1:
            count_mult += 1
            # print("Ambiguous matches: COREF {}".format(coref))
            # print("Texts: ", texts)
            # print("Matches: ")
            # for x in set(match_ids):
            #     print("\t", x, charInfo['id2names'][x])
            # print()
            rows.append([coref, match_ids, -1])

        if len(set(match_ids))==0:
            count_none += 1
            # print("No matches: COREF {}".format(coref))
            # print("Texts: ", texts)
            
            # print()
            rows.append([coref, match_ids, -1])

        if len(set(match_ids))==1:
            count_match += 1
            # print("Matched: COREF {}".format(coref))
            # print("Texts: ", texts)
            # for x in set(match_ids):
            #     print("\t", x, charInfo['id2names'][x])
            # print()

            match = match_ids[0]
            rows.append([coref, match_ids, match])
        
    print("Multiple match count: {}/{}".format(count_mult, len(coref2texts)))
    print("No match count: {}/{}".format(count_none, len(coref2texts)))
    print("Unique match count: {}/{}".format(count_match, len(coref2texts)))

    matchdf = pd.DataFrame(rows, columns=['COREF', 'matches', 'pdncID'])

    matchdf.to_csv(os.path.join(SAVE_ROOT, novel, 'coref_matches.csv'), index=False)
    matchdf.set_index('COREF', inplace=True)

    entdf['pdncID'] = [matchdf.loc[x]['pdncID'] for x in entdf.index]

    matched_ents = entdf[entdf['pdncID']!=-1]
    matched_ents.to_csv(os.path.join(SAVE_ROOT, novel, 'booknlp_matched_ents.csv'))

    return matched_ents


def make_array(s):
    if isinstance(s, str):
        ev = eval(s)
        if isinstance(ev, list):
            return ev
        
        return [s]
    if isinstance(s, list):
        return s
    return [s]

def get_offset_bytes(qtext, sb, eb):
	i = 0
	while(qtext[i] in string.whitespace):
		i += 1
	sb = sb + i

	i = 0
	while (qtext[-(i+1)] in string.whitespace):
		i += 1
	eb = eb - i
	return sb, eb


def pdnc_process_novel(novel):
    if not os.path.isdir(os.path.join(SAVE_ROOT, novel)):
        os.mkdir(os.path.join(SAVE_ROOT, novel))

    qdf = read_quote_df(novel)
    ntext = get_ntext(novel)
    charInfo = read_char_info(novel)
    # df = split_qdf(qdf)

    mrows = []
    for _, row in qdf.iterrows():
        qid = row['qID']
        mts = make_array(row['menTexts'])
        mets = make_array(row['menEnts'])
        msps = make_array(row['menSpans'])
        
        for mt, me, ms in zip(mts, mets, msps):
            assert ntext[ms[0]:ms[1]] == mt
            try:
                mids = [charInfo['name2id'][x.strip()] for x in me]
            except KeyError as e:
                mids = []
            if len(mids) == 1:
                mrows.append([qid, mt, ms[0], ms[1], mids, mids[0]])
            else:
                mrows.append([qid, mt, ms[0], ms[1], mids, -1])

    mdf = pd.DataFrame(mrows, columns=['qID', 'menText', 'start_byte', 'end_byte', 'matches', 'pdncID'])

    mdf.to_csv(os.path.join(SAVE_ROOT, novel, 'pdncMentions.csv'), index=False)

    mdf = mdf[mdf['pdncID']!=-1]

    mdf.to_csv(os.path.join(SAVE_ROOT, novel, 'pdnc_single_mentions.csv'), index=False)

    return mdf

def replace_single(text):
    
    text = re.sub('(?<![\r\n])(\r?\n|\n?\r)(?![\r\n])'," ", text) #replace single newlines

    return text

def explicit_process_novel(novel):
    if not os.path.isdir(os.path.join(SAVE_ROOT, novel)):
        os.mkdir(os.path.join(SAVE_ROOT, novel))

    charInfo = read_char_info(novel)
    ntext = get_ntext(novel)
    rtext = ntext.lower()
   
    e_name2id = get_enhanced_char_list(charInfo['name2id'])
    charNames = set([x.lower() for x in list(e_name2id.keys())])
    search_names = sorted(charNames, key=lambda x: len(x), reverse=True)
  
    exists = set()

    erows = []
    for name in search_names:
        cid = e_name2id[name]

        matches = list(re.finditer(rf'\b{name}\b', rtext))
        for m in matches:
            s, e = m.start(), m.end()
            if (s not in exists) and (e not in exists):
                mt = ntext[s:e]
                erows.append([name, mt, s, e, cid])
                for c in range(s, e):
                    exists.add(c)

    edf = pd.DataFrame(erows, columns=['exp_name', 'text', 'start_byte', 'end_byte', 'pdncID'])
    edf.to_csv(os.path.join(SAVE_ROOT, novel, 'explicit_mentions.csv'), index=False)
    return edf


    
def read_booknlp_mens(novel):
    df = pd.read_csv(os.path.join(SAVE_ROOT, novel, 'booknlp_matched_ents.csv'))
    return df

def read_pdnc_mens(novel):
    df = pd.read_csv(os.path.join(SAVE_ROOT, novel, 'pdnc_single_mentions.csv'))
    return df

def read_exp_mens(novel):
    df = pd.read_csv(os.path.join(SAVE_ROOT, novel, 'explicit_mentions.csv'))
    return df

def is_conflict(exists, s, e):
    for c in range(s, e):
        if c in exists:
            return True
    return False

def merge_mentions(novel):
    print(novel)
    
    bookdf = read_booknlp_mens(novel)
    # pdncdf = read_pdnc_mens(novel)
    expdf = read_exp_mens(novel)

    exists = set()
    rows = []

#     name = 'pdnc'
#     idCol = 'qID'
#     count = 0
#     for _, row in pdncdf.iterrows():
#         s, e = row['start_byte'], row['end_byte']
#         mt = row['menText']
#         cid = row['pdncID']
#         iden = row[idCol]
#         if not is_conflict(exists, s, e):
#             rows.append([name, idCol, iden, mt, s, e, cid])
#             count += 1
#             for c in range(s, e):
#                 exists.add(c)
#     print("Added {}/{} from {}".format(count, len(pdncdf), name))
                
    name = 'exp'
    idCol = 'exp_name'
    count = 0
    for _, row in expdf.iterrows():
        s, e = row['start_byte'], row['end_byte']
        mt = row['text']
        cid = row['pdncID']
        iden = row[idCol]
        if not is_conflict(exists, s, e):
            rows.append([name, idCol, iden, mt, s, e, cid])
            count += 1
            for c in range(s, e):
                exists.add(c)
    print("Added {}/{} from {}".format(count, len(expdf), name))

    name = 'booknlp'
    idCol = 'COREF'
    count = 0
    for _, row in bookdf.iterrows():
        s, e = row['start_byte'], row['end_byte']
        mt = row['text']
        cid = row['pdncID']
        iden = row[idCol]
        if not is_conflict(exists, s, e):
            rows.append([name, idCol, iden, mt, s, e, cid])
            count += 1
            for c in range(s, e):
                exists.add(c)
    print("Added {}/{} from {}".format(count, len(bookdf), name))

    alldf = pd.DataFrame(rows, columns=['source', 'idColName', 'iden', 'text', 'startByte', 'endByte', 'pdncID'])
    alldf.sort_values(by='startByte', inplace=True)
    alldf.to_csv(os.path.join(SAVE_ROOT, novel, 'merged_mentions.csv'), index=False)
    return alldf

def read_men_df(novel):
    df = pd.read_csv(os.path.join(SAVE_ROOT, novel, 'merged_mentions.csv'))
    return df

def check_conflicts(novel):
    print(novel)
    merged_df = read_men_df(novel)
    merged_df.sort_values(by='startByte', inplace=True)
    to_delete = []
    missed = []
    for i in range(len(merged_df)-1):

        if merged_df.iloc[i]['endByte'] > merged_df.iloc[i+1]['startByte']:
            row1 = merged_df.iloc[i]
            row2 = merged_df.iloc[i+1]

            source1, source2 = row1['source'], row2['source']
            start1, end1 = row1['startByte'], row1['endByte']
            start2, end2 = row2['startByte'], row2['endByte']

            if (source1 in ['pdnc', 'exp']) and (source2 not in ['pdnc', 'exp']):
                to_delete.append(i+1)

            elif (source2 in ['pdnc', 'exp']) and (source1 not in ['pdnc', 'exp']):
                to_delete.append(i)

            else:
                #pick the shorter
                len1 = len(row1['text'])
                len2 = len(row2['text'])

                if len1 >= len2:
                    to_delete.append(i)
                else:
                    to_delete.append(i+1)
            missed.append(i)
    print(to_delete, missed)
    
    
    
for novel in novels:
    booknlp_process_novel(novel)
    # pdnc_process_novel(novel)
    explicit_process_novel(novel)
#MERGE
    merge_mentions(novel)