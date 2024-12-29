import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))

from booknlpen.english.speaker_attribution import BERTSpeakerID
import torch.nn as nn
import torch
import argparse
import json, re, sys, string
from collections import Counter, defaultdict
import csv
from random import shuffle
import sys
import pandas as pd
import numpy as np
import pickle as pkl
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
import torch.nn.functional as F 
#import pytorch_warmup as warmup
from sentence_transformers import SentenceTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)


def get_pairs(labels):
    pos = torch.where(labels == 1 )[0]
    neg = torch.where(labels ==0 )[0]
    x = []
    y = []
    for p in pos : 
        x.append(p.repeat(len(neg)))
        y.append(neg)
    return torch.cat(x), torch.cat(y), torch.FloatTensor([1] * torch.cat(x).size(0)).to(labels.device)

def read_speaker_data(filename, shuffle_data=True, split = "train"):

    with open(filename) as file:

        data={}

        for line in file:
            cols=line.rstrip().split("\t")

            sid=cols[0]
            qid = cols[1]
            chapID = cols[2]
            eid=cols[3]
            cands=json.loads(cols[5])
            try :
                quote=int(cols[4]) #quote token position
            except: 
                quote=json.loads(cols[4]) # case where quote is a (start, end) tuple
            text=cols[6].split(" ")
            is_explicit=cols[7]

            for s,e,_,_ in cands:
                #start and end token index in context
                if s > len(text) or e > len(text):
                    print("reading problem", s, e, len(text))
                    sys.exit(1)
            all_mentions = json.loads(cols[-1])
            if sid not in data:
                data[sid]=[]
            if split == "train" : 
                cands_eid = [i[-1] for i in cands]
                if eid in cands_eid : 
                    # skip examples where true speaker not in candidates 
                    data[sid].append((qid, chapID, eid, cands, quote, text, is_explicit, all_mentions))
            else : 
                data[sid].append((qid, chapID, eid, cands, quote, text, is_explicit, all_mentions))

        x=[]
        m=[]
        o=[]

        sids=list(data.keys())

        #shuffle(sids)

        grouped_qids = defaultdict(list)
        for sid in sids:			
            for qid, chapID, eid, cands, quote, text, is_explicit, all_mentions in data[sid]:
                x.append(text)
                m.append((eid, cands, quote, all_mentions))
                o.append((sid, qid, chapID, is_explicit))
                grouped_qids[sid].append(qid)

        # shuffling data
        if shuffle_data : 
            indices = list(range(len(x)))
            np.random.shuffle(indices)
            print("First 10 indices: ", indices[:10])
            x = [x[i] for i in indices]
            m = [m[i] for i in indices]
            o = [o[i] for i in indices]
    return grouped_qids, x, m, o


def read_explicit_quotes_by_speaker(novel_ids, source_path) : 
    data = {}
    for nid, qids in novel_ids.items():
        novel_data = {}
        quote_path = os.path.join(source_path, nid, "quote_info.csv")
        quote_df = pd.read_csv(quote_path)

        char_info_path = os.path.join(source_path, nid, "charInfo.dict.pkl")
        char_info = pkl.load(open(char_info_path, "rb"))
        for eid in char_info["id2names"].keys() :
            novel_data[eid] = []
            if quote_df[quote_df["speakerID"]==eid].shape[0] > 0: 	
                if quote_df[quote_df["speakerID"]==eid]["speakerType"].iloc[0] != "minor" : 
                    sub = quote_df[(quote_df["speakerID"]==eid) & (quote_df["qType"]=="Explicit")]
                    if sub.shape[0]>0 :
                        # Only take quotes from given split
                        sub = sub[sub["qID"].isin(qids)]
                        novel_data[eid] = sub["qText"].tolist()
        data[nid] = novel_data
    return data

def read_chapterwise_explicit_quotes_by_speaker(novel_ids, source_path) : 
    data = {}
    for nid, qids in novel_ids.items():
        novel_data = {}
        quote_path = os.path.join(source_path, nid, "quote_info.csv")
        quote_df = pd.read_csv(quote_path)

        char_info_path = os.path.join(source_path, nid, "charInfo.dict.pkl")
        char_info = pkl.load(open(char_info_path, "rb"))
        for eid in char_info["id2names"].keys() :

            novel_data[eid] = {k:[] for k in quote_df["startChapID"].unique()}
            sub = quote_df[quote_df["speakerID"]==eid]
            if sub.shape[0] > 0:
                if sub["speakerType"].iloc[0] != "minor" : 
                    for chapId in sub["startChapID"].unique():
                        subsub = sub[(sub["qType"]=="Explicit") & (sub["startChapID"]==chapId)]
                        if subsub.shape[0]>0 :
                            # Only take quotes from given split
                            novel_data[eid][chapId] = subsub[subsub["qID"].isin(qids)]["qText"].tolist()
        data[nid] = novel_data
    return data

def tokenize(tokenizer, quotes, batch_first = False, max_length=64) : 
    tokens = tokenizer(quotes, max_length = max_length, return_tensors = "pt", truncation=True, padding="max_length")

    if not batch_first:  
        tokens["input_ids"] = tokens["input_ids"].reshape(1, -1, max_length)
        tokens["attention_mask"] = tokens["attention_mask"].reshape(1, -1, max_length)
    else : 
        tokens["input_ids"] = tokens["input_ids"].reshape(-1, 1, max_length)
        tokens["attention_mask"] = tokens["attention_mask"].reshape(-1, 1, max_length)

    return tokens


def embed_luar_quotes(quotes_by_novel, model, tokenizer, max_n_quotes=None, model_type = "luar") : 
    data = {}
    try : 
        size = model.config.embedding_size 
    except : 
        size = model.get_sentence_embedding_dimension()
    with torch.no_grad() : 
        for nid in quotes_by_novel : 
            novel_data = {}
            for eid, quotes in quotes_by_novel[nid].items() : 
                if len(quotes) == 0:	
                    # tokens = tokenize(tokenizer, quotes=[""])
                    novel_data["CHAR_" + str(eid)] = torch.zeros(1,size)
                else : 
                    if max_n_quotes is not None : 
                        idx = np.random.permutation(len(quotes))[:max_n_quotes]
                        quotes = [quotes[id] for id in idx]
                    if model_type == "luar": 
                        tokens = tokenize(tokenizer, quotes, batch_first=False, max_length=64)
                        out =  model(**tokens.to(device)).cpu()
                    else : 
                        out = model.encode(quotes, device=model.device, normalize_embeddings=True, convert_to_numpy=False, convert_to_tensor=True).cpu().mean(0).unsqueeze(0)
                    novel_data["CHAR_" + str(eid)] = out
            data[nid] = novel_data
    return data

@torch.no_grad()
def embed_quotes(source_path, data, model, tokenizer, batch_size=32, model_type="luar") :
    novel_data = {}
    # Load quote dfs
    for nid in set([d[0] for d in data]) : 
        quote_path = os.path.join(source_path, nid, "quote_info.csv")
        novel_data[nid] = pd.read_csv(quote_path)
    quotes = []
    for sid, qid, _, _ in data: 
        quotes.append(novel_data[sid][novel_data[sid]["qID"] == qid]["qText"].values.item())
    if model_type == "luar" : 
        quote_data = [] 
        for bid in range(0,len(quotes), batch_size): 
            encodings = tokenizer(quotes[bid:bid+batch_size], max_length=64, truncation=True, padding="max_length", return_tensors="pt").to(device)
            encodings = {k:v.unsqueeze(1) for k,v in encodings.items()}
            quote_data.append(model(**encodings).cpu())
        return torch.cat(quote_data)
    else : 
        quote_data = model.encode(quotes, device=model.device, normalize_embeddings=True, convert_to_numpy=False, convert_to_tensor=True).cpu()
        return quote_data
    # encodings = tokenizer.batch_encode_plus(batches, max_length=64, truncation=True, padding="max_length")


def embed_luar_global_quotes(quotes_by_novel, model, tokenizer, max_n_quotes=None, model_type = "luar") :
    try :
        size = model.config.embedding_size 
    except : 
        size = model.get_sentence_embedding_dimension()
    with torch.no_grad() : 
        data = {}
        for nid in quotes_by_novel : 
            novel_data = {f"CHAR_{eid}":{} for eid in quotes_by_novel[nid].keys()}
            for eid, quotes_by_chapter in quotes_by_novel[nid].items() : 
                char_data = []
                chapters = list(quotes_by_chapter.keys())
                for cid in chapters : 
                    quotes = quotes_by_chapter[cid]
                    if len(quotes) != 0 :
                        # take most recent quotes
                        if max_n_quotes is not None : 
                            idx = np.random.permutation(len(quotes))[:max_n_quotes]
                            quotes = [quotes[id] for id in idx]
                        if model_type == "luar" : 
                            tokens = tokenize(tokenizer, quotes, batch_first=False, max_length=64)
                            out =  model(**tokens.to(device)).cpu()
                        else : 
                            out = model.encode(quotes, device=model.device, normalize_embeddings=True, convert_to_numpy=False, convert_to_tensor=True).cpu().mean(0).unsqueeze(0)
                        char_data.append(out)
                        #novel_data["CHAR_" + str(eid)][chapters[cid]] = model(**tokens.to(device)).cpu()
                if len(char_data) > 0 :
                    novel_data["CHAR_" + str(eid)] = torch.stack(char_data).mean(0)
                else : 
                    novel_data["CHAR_" + str(eid)] = torch.zeros(1,size)
            data[nid] = novel_data
    return data

def embed_luar_chapterwise_quotes(quotes_by_novel, model, tokenizer, max_n_quotes=None, window="left", model_type = "luar") : 
    try :
        window = int(window)
    except : 
        pass
    try :
        size = model.config.embedding_size 
    except : 
        size = model.get_sentence_embedding_dimension()
    with torch.no_grad() : 
        data = {}
        for nid in tqdm(quotes_by_novel) : 
            novel_data = {f"CHAR_{eid}":{} for eid in quotes_by_novel[nid].keys()}
            for eid, quotes_by_chapter in quotes_by_novel[nid].items() : 
                chapters = list(quotes_by_chapter.keys())
                for cid in range(len(quotes_by_chapter)) : 
                    if window == "left" : 
                        chapter_window = [i for i in range(cid+1) if i >= 0]
                    elif isinstance(window, int) : 
                        chapter_window = [i for i in range(cid - window, cid+1) if i >= 0]
                    quotes = []
                    for cidd in chapter_window : 
                        if cidd >= 0 :
                            idx = chapters[cidd]
                            quotes.extend(quotes_by_chapter[idx])
                    if len(quotes) == 0:
                        # tokens = tokenize(tokenizer, quotes=[""])
                        novel_data["CHAR_" + str(eid)][chapters[cid]] = torch.zeros(1,size)
                    else :
                        #idx = np.random.permutation(len(quotes))[:max_n_quotes]
                        # take most recent quotes
                        if max_n_quotes is not None : 
                            quotes = quotes[-max_n_quotes:]
                        if model_type == "luar": 
                            tokens = tokenize(tokenizer, quotes, batch_first=False, max_length=64)
                            out =  model(**tokens.to(device)).cpu()
                        else : 
                            out = model.encode(quotes, device=model.device, normalize_embeddings=True, convert_to_numpy=False, convert_to_tensor=True).cpu().mean(0).unsqueeze(0)
                        novel_data["CHAR_" + str(eid)][chapters[cid]] = out
                    data[nid] = novel_data
    return data


def predict(model, test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches):
    model.eval()
    gold_eids = []
    pred_eids = []
    meta_info = []
    pred_confs = []

    with torch.no_grad():
        idd = 0
        for x1, m1, y1, o1, i1 in zip(test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches):
            y_pred = model.forward(x1, m1)
            predictions=torch.argmax(y_pred, axis=1).detach().cpu().numpy()
            orig, meta = o1
            for idx, pred in enumerate(predictions):

                prediction = pred[0]

    #                 sent=orig[idx]
                pred_conf = y_pred[idx][prediction].detach().cpu().numpy()[0]

                # if prediction >= len(meta[idx][1]):
                # 	prediction=torch.argmax(y_pred[idx][:len(meta[idx][1])])
                # 	pred_conf = y_pred[idx][prediction].detach().cpu().numpy()[0]

                gold_eids.append(y1["quote_eids"][idx])

                predval=y1["eid"][idx][prediction]
                if predval is None:
                    predval="none-%s" % (idd)
                pred_eids.append(predval)
                pred_confs.append(pred_conf)

                meta_info.append(i1[idx])
                idd += 1

    return gold_eids, pred_eids, pred_confs, meta_info


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', help='Filename containing training data', required=False)
    # parser.add_argument('--trainData', help='Filename containing training data', required=False)
    # parser.add_argument('--devData', help='Filename containing dev data', required=False)
    # parser.add_argument('--testData', help='Filename containing test data', required=False)
    parser.add_argument('--sourceData', help='Filename containing source data', required=False)
    parser.add_argument('--base_model', help='Base BERT model', required=False)
    parser.add_argument('--savePath', help='Folder to save outputs', required=False)
    parser.add_argument('--max_n_quotes', help='Number of quotes to be used by LUAR', required=False, type=int, default=None)
    parser.add_argument('--chapterwise', help='whether to use chapterwise luar embeddings', required=False, action="store_true", default=False)
    parser.add_argument('--chapter_window', help='# of chapters to use for luar emb', required=False, default="left")
    parser.add_argument('--model_mode', help='which features to use in model', required=False, default="vanilla")
    parser.add_argument('--unmask_quotes', help='whether to use chapterwise luar embeddings', required=False, action="store_true", default=False)
    parser.add_argument('--gcp', help='whether to use gradient checkpointing', required=False, action="store_true", default=False)
    parser.add_argument('--use_amp', help='whether to use fp16 training', required=False, action="store_true", default=False)
    parser.add_argument('--path_to_ckpt', help='path to drama luar checkpoint', required=False, default=None, type=str)
    parser.add_argument('--num_cands', help='numbr of candidate mentions', default=10, type=int)

    args = vars(parser.parse_args())
    print(parser.parse_args())
    # trainData=args["trainData"]
    # devData=args["devData"]
    # testData=args["testData"]
    dataPath = args["dataPath"]
    base_model=args["base_model"]
    savePath=args["savePath"]
    sourceData=args["sourceData"]
    chapterwise = args["chapterwise"]
    chapter_window = args["chapter_window"]
    mode = args["model_mode"]
    unmask_quotes=args["unmask_quotes"]
    gcp=args["gcp"]
    use_amp=args["use_amp"]
    max_n_quotes=args["max_n_quotes"]
    path_to_ckpt = args["path_to_ckpt"]
    num_cands = args["num_cands"]

    config = {
        "embedding_size":512,
        "model_name" : "sentence-transformers/all-distilroberta-v1", 
        "gradient_checkpointing": False
        }
    class ModelArgument : 
        embedding_size = 1024
        model_name = "sentence-transformers/all-mpnet-base-v2"
        gradient_checkpointing = False
        def __init__(self, config) : 
            for key,val in config.items() :
                if hasattr(self, key) :
                    setattr(self, key, val)         
    model_args = ModelArgument(config)

    for split in range(5) : 
        print(f"Starting processing split {split}")
        trData = os.path.join(dataPath, f"split_{split}", "quotes.train.txt")
        deData = os.path.join(dataPath, f"split_{split}", "quotes.dev.txt")
        teData = os.path.join(dataPath, f"split_{split}",  "quotes.test.txt")
        save = os.path.join(savePath, f"split_{split}")
        os.makedirs(save, exist_ok=True)

        model_name = os.path.join(save, 'best_model.model')
        print("Reading data..")
        train_ids, train_x, train_m, train_i=read_speaker_data(trData, split="train")
        dev_ids, dev_x, dev_m, dev_i=read_speaker_data(deData, shuffle_data=False, split="dev")
        test_ids, test_x, test_m, test_i=read_speaker_data(teData, shuffle_data=False, split="test")

        if mode == "uar_scene" :
            tokenizer = AutoTokenizer.from_pretrained("gasmichel/UAR_scene", trust_remote_code=True)
            model = AutoModel.from_pretrained("gasmichel/UAR_scene", trust_remote_code=True).to(device)
            model.eval()
            model=model.to(device)
            model_type = "luar"
        elif mode == "uar_play" :
            tokenizer = AutoTokenizer.from_pretrained("gasmichel/UAR_Play", trust_remote_code=True)
            model = AutoModel.from_pretrained("gasmichel/UAR_Play", trust_remote_code=True).to(device)
            model.eval()
            model=model.to(device)
            model_type = "luar"
        elif "luar" in mode : 
            tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
            model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True).to(device)
            model.eval()
            model_type = "luar"
        elif "semantics" in mode : 
            model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            model.max_seq_length = 64
            model.eval()
            model_type = "semantics"
            tokenizer = None 
        if mode != "vanilla" : 
            print("Character embeddings..")
            tr_quote_emb = embed_quotes(sourceData, train_i, model,tokenizer, model_type=model_type)
            val_quote_emb = embed_quotes(sourceData, dev_i, model,tokenizer, model_type=model_type)
            test_quote_emb = embed_quotes(sourceData, test_i, model,tokenizer, model_type=model_type)
            if not chapterwise : 
                train_quotes_by_char = read_explicit_quotes_by_speaker(train_ids, sourceData)
                # #dev_quotes_by_char = read_explicit_quotes_by_speaker(dev_ids, sourceData)
                test_quotes_by_char = read_explicit_quotes_by_speaker(test_ids, sourceData)
                train_style_emb = embed_luar_quotes(train_quotes_by_char, model, tokenizer, max_n_quotes=max_n_quotes, model_type=model_type)
                test_style_emb = embed_luar_quotes(test_quotes_by_char, model, tokenizer, max_n_quotes=max_n_quotes, model_type=model_type)

                # train_quotes_by_char = read_chapterwise_explicit_quotes_by_speaker(train_ids, sourceData)
                # test_quotes_by_char = read_chapterwise_explicit_quotes_by_speaker(test_ids, sourceData)
                # print("LUAR embeddings..")
                # train_style_emb = embed_luar_global_quotes(train_quotes_by_char, model, tokenizer, max_n_quotes=max_n_quotes)
                # test_style_emb = embed_luar_global_quotes(test_quotes_by_char, model, tokenizer, max_n_quotes=max_n_quotes)
            else :
                train_quotes_by_char = read_chapterwise_explicit_quotes_by_speaker(train_ids, sourceData)
                #dev_quotes_by_char = read_chapterwise_explicit_quotes_by_speaker(dev_ids, sourceData)
                test_quotes_by_char = read_chapterwise_explicit_quotes_by_speaker(test_ids, sourceData)
                train_style_emb = embed_luar_chapterwise_quotes(train_quotes_by_char, model, tokenizer, max_n_quotes=max_n_quotes, window=chapter_window, model_type=model_type)
                test_style_emb = embed_luar_chapterwise_quotes(test_quotes_by_char, model, tokenizer, max_n_quotes=max_n_quotes, window=chapter_window, model_type=model_type)
        else : 
            train_style_emb = test_style_emb = None
            tr_quote_emb = val_quote_emb = test_quote_emb = None

        print("Getting batches..")
        #populate assigned speakers from training data
        metric="accuracy"

        bertSpeaker=BERTSpeakerID(base_model=base_model, mode=mode, unmask_quote=unmask_quotes)
        if gcp: 
            bertSpeaker.bert.gradient_checkpointing_enable()
        bertSpeaker.to(device)
        train_x_batches, train_m_batches, train_y_batches, train_o_batches, train_i_batches=bertSpeaker.get_batches(train_x, train_m, train_i, train_style_emb, tr_quote_emb, batch_size=16, by_chapter=chapterwise, num_cands=num_cands)
        # For dev set, we use same luar embeddings as train as these are quotes from same novels
        dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, dev_i_batches=bertSpeaker.get_batches(dev_x, dev_m, dev_i, train_style_emb, val_quote_emb, batch_size=16, by_chapter=chapterwise, use_all_mentions=True)#, num_cands=num_cands)
        test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches=bertSpeaker.get_batches(test_x, test_m, test_i, test_style_emb, test_quote_emb, batch_size=16, by_chapter=chapterwise, use_all_mentions=True)# num_cands=num_cands)
        print("Training..")
        patience = 5

        bert_params = bertSpeaker.bert.parameters()
        task_params = [p for n,p in bertSpeaker.named_parameters() if "bert" not in n]

        #optimizer = torch.optim.AdamW([
        #	{"params" : bert_params, "lr":5e-6},
        #	{"params": task_params}], lr=1e-4, weight_decay = 1e-3)

        optimizer = torch.optim.AdamW(bertSpeaker.parameters(), lr=5e-6)
        best_dev_acc = 0.

        num_epochs=20
        num_steps = len(train_m_batches) * num_epochs
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        # # warmup learnign rate over 2 epochs
        # warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=2 * len(train_m_batches))
        # margin_loss = nn.MarginRankingLoss(margin=1.0)
        # margin_loss = nn.MarginRankingLoss(margin=1.0)

        scaler = GradScaler(enabled=use_amp)
        accumulation_steps = 1
        pat_count = 0
        for epoch in range(num_epochs):
            count = 0
            bertSpeaker.train()
            bigLoss=0

            for i, (x1, m1, y1) in tqdm(enumerate(zip(train_x_batches, train_m_batches, train_y_batches)), desc = f"SPLIT {split} - Epoch {epoch}", total=len(train_x_batches)):
                with autocast(enabled=use_amp):
                    y_pred = bertSpeaker.forward(x1, m1)

                    batch_y=y1["y"].unsqueeze(-1)
                    batch_y=torch.abs(batch_y-1)*-100
                    #NICE  #all true candidates are accepted
                    true_preds=y_pred+batch_y

                    golds_sum=torch.logsumexp(true_preds, 1)
                    all_sum=torch.logsumexp(y_pred, 1)

                    loss=torch.sum(all_sum-golds_sum)
# 					loss =0 
# 					for idx in range(len(y_pred)) : 
# 						pos, neg, labels = get_pairs(y1["y"][idx])
# 						# print(margin_loss(y_pred[idx][pos].squeeze(), y_pred[idx][neg].squeeze(), labels))

# 						idx_loss = margin_loss(y_pred[idx][pos].squeeze(), y_pred[idx][neg].squeeze(), labels)
# 						if not torch.isnan(idx_loss).item() : 
# 							loss += idx_loss
                        # scaler.scale(loss).backward()

                scaler.scale(loss).backward()
                bigLoss+=loss.detach().cpu().item()
                count += len(x1)

                #loss.backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)  # Update parameter
                    scaler.update()
                    # optimizer.step()
                    optimizer.zero_grad()
# 				with warmup_scheduler.dampening():
# 					lr_scheduler.step()

            print("\t\t\tEpoch %s loss: %.3f" % (epoch, bigLoss / count))

            # Evaluate; save the model that performs best on the dev data
            dev_F1, dev_acc, dev_F1_noexp, dev_acc_noexp =bertSpeaker.evaluate(dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, dev_i_batches, epoch)
            # dev_F1, dev_acc, dev_F1_noexp, dev_acc_noexp =bertSpeaker.evaluate(test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches, epoch)
            # if epoch % 2 == 0 :
            # 	_, _, _, _ =bertSpeaker.evaluate(train_x_batches, train_m_batches, train_y_batches, train_o_batches, train_i_batches, "TRAIN")

            sys.stdout.flush()
            if epoch % 1 == 0:
                # metric is dev F1 on non-explicits
                if dev_acc_noexp > best_dev_acc:
                    pat_count = 0  
                    torch.save(bertSpeaker.state_dict(), model_name)
                    best_dev_acc = dev_acc_noexp
                else : 
                    pat_count +=1 
            if pat_count == patience:
                break 
        # Test with best performing model on dev
        bertSpeaker.load_state_dict(torch.load(model_name, map_location=device))
        bertSpeaker.eval()

        test_F1, test_acc, test_F1_noexp, test_acc_noexp=bertSpeaker.evaluate(test_x_batches, test_m_batches, test_y_batches, test_o_batches, test_i_batches, "TEST")

        print("\n[ALL] Test F1:\t%.3f\t, Test accuracy:\t%.3f" % (test_F1, test_acc))
        print("[NOEXP] Test F1:\t%.3f\t, Test accuracy:\t%.3f\n" % (test_F1_noexp, test_acc_noexp))
        with open(os.path.join(save, 'test_results.pkl'), "wb") as f :
            pkl.dump({
                "f1" : test_F1,
                "acc" : test_acc,
                "nexp_f1" : test_F1_noexp,
                "nexp_acc" : test_acc_noexp
            }, f)
        gold_eids, pred_eids, pred_confs, meta_info = predict(bertSpeaker, dev_x_batches, dev_m_batches, dev_y_batches, \
                                            dev_o_batches, dev_i_batches)

        # resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

        with open(os.path.join(save, 'val_preds_max.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["novel_id", "q_id", "is_explicit", "gold", "pred", "confidence"])
            for meta, gold, pred, conf in zip(meta_info, gold_eids, pred_eids, pred_confs):
                writer.writerow([meta[0], meta[1], meta[-1], gold, pred, conf])

        gold_eids, pred_eids, pred_confs, meta_info = predict(bertSpeaker, test_x_batches, test_m_batches, test_y_batches, \
                                            test_o_batches, test_i_batches)
        # resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

        with open(os.path.join(save, 'test_preds_max.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["novel_id", "q_id", "is_explicit", "gold", "pred", "confidence"])
            for meta, gold, pred, conf in zip(meta_info, gold_eids, pred_eids, pred_confs):
                writer.writerow([meta[0], meta[1], meta[-1], gold, pred, conf])

        # remove model
        #os.remove(model_name)


        # gold_eids, pred_eids, pred_confs, meta_info = predict(bertSpeaker, dev_x_batches, dev_m_batches, dev_y_batches, \
        # 									dev_o_batches, dev_i_batches)

        # resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

        # with open(os.path.join(save, 'val_preds.csv'), 'w') as f:
        # 	writer = csv.writer(f)
        # 	writer.writerow(["novel_id", "q_id", "is_explicit", "gold", "pred", "confidence"])
        # 	for meta, gold, pred, conf in zip(meta_info, gold_eids, resolved_pred_eids, pred_confs):
        # 		writer.writerow([meta[0], meta[1], meta[-1], gold, pred, conf])


        # gold_eids, pred_eids, pred_confs, meta_info = predict(bertSpeaker, test_x_batches, test_m_batches, test_y_batches, \
        # 									test_o_batches, test_i_batches)
        # resolved_pred_eids = resolve_sequential(pred_eids, meta_info, assigned_speakers)

        # with open(os.path.join(save, 'test_preds.csv'), 'w') as f:
        # 	writer = csv.writer(f)
        # 	writer.writerow(["novel_id", "q_id", "is_explicit", "gold", "pred", "confidence"])
        # 	for meta, gold, pred, conf in zip(meta_info, gold_eids, resolved_pred_eids, pred_confs):
        # 		writer.writerow([meta[0], meta[1], meta[-1], gold, pred, conf])					

































































































































