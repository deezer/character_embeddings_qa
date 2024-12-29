import sys
import re
import random
from random import shuffle
from math import sqrt, exp, isnan
from transformers import BertTokenizer, BertModel, AutoConfig, AutoModel
import torch.nn as nn
import torch
import numpy as np
import argparse
import json
from booknlpen.common.b3 import b3
import torch.nn.functional as F 
from collections import Counter
from tqdm import tqdm

PINK = '\033[95m'
ENDC = '\033[0m'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTSpeakerID(nn.Module):

    def __init__(self, base_model=None, mode="vanilla", unmask_quote=False):
        "args: mode accepts values ['vanilla', 'char_emb', 'char_and_luar_emb']"
        assert mode in ['vanilla', 'char_emb', 'char_and_luar_emb', "drama_luar", "semantics"], "arg: mode accepts values ['vanilla', 'char_emb', 'char_and_luar_emb', 'drama_luar']"
        super().__init__()
        modelName = base_model
        # modelName=base_model.split("/")[-1]
        # modelName=re.sub("(.)*_bert_", "bert_", modelName)
        # modelName=re.sub("-v\d.*$", "", modelName)
        #og
        # modelName=re.sub("^speaker_", "", modelName)
        # modelName=re.sub("-v\d.*$", "", modelName)

        # matcher=re.search(".*-(\d+)_H-(\d+)_A-.*", modelName)
        # bert_dim=0
        # modelSize=0
        # self.num_layers=0
        # if matcher is not None:
        # 	bert_dim=int(matcher.group(2))
        # 	self.num_layers=min(4, int(matcher.group(1)))

        # 	modelSize=self.num_layers*bert_dim

        # assert bert_dim != 0
        self.mode = mode
        # tokPath = base_model

        self.tokenizer = BertTokenizer.from_pretrained(modelName, do_lower_case=False, do_basic_tokenize=False)
        if not unmask_quote:
            self.tokenizer.add_tokens(["[QUOTE]", "[ALTQUOTE]", "[PAR]"], special_tokens=True)
        else : 
            self.tokenizer.add_tokens(["[ALTQUOTE]", "[PAR]"], special_tokens=True)

        # config = AutoConfig.from_pretrained(modelName)
        # self.bert = AutoModel.from_config(config)
        self.bert = BertModel.from_pretrained(modelName)
        self.bert_dim = self.bert.config.hidden_size
        self.bert.resize_token_embeddings(len(self.tokenizer))
        #self.char_emb = nn.Embedding(200, 512, padding_idx=0)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        # if unmask_quote : 
        # 	quote_dim = bert_dim
        # else : 
        # 	quote_dim = bert_dim
        # if self.mode == "vanilla":
        # 	self.fc = nn.Linear(bert_dim + quote_dim, 100)
        # elif self.mode == "char_emb" : 
        # 	self.fc = nn.Linear(bert_dim + quote_dim + 512, 100)
        self.unmask_quote = unmask_quote
        if ("luar" in self.mode):#& ("drama" not in self.mode):
            style_dim = 512
        # elif ("luar" in self.mode) & ("drama" in self.mode) : 
        #     style_dim = 1024
        elif "semantics" in self.mode:
            style_dim = 768
        else :
            style_dim = 0
            # self.fc = nn.Linear(bert_dim + quote_dim + 2*512, 100)
            # self.fc = nn.Linear(bert_dim + quote_dim + 512, 100)
        # if "luar" in self.mode : 
        # 	# if self.unmask_quote:
        # 	# 	self.quote_fc = nn.Linear(self.bert_dim * 2 + style_dim, 256)
        # 		# 
        # 	if self.bert_dim != style_dim : 
        # 		self.aligner = nn.Linear(style_dim, bert_dim)

        # 	self.fc = nn.Linear(bert_dim * 4, 100)

        # 	# self.fc = nn.Linear(128 * 2, 64)
        # else : 
        # 	if self.unmask_quote:
        # 		self.fc = nn.Linear(self.bert_dim * 4, 64)
        # 	else :
        # 		self.fc = nn.Linear(self.bert_dim * 3, 64)
        if not self.unmask_quote :
            if ("luar" in self.mode) or ("semantics" in self.mode): 

                self.fc = nn.Linear(self.bert_dim * 3 + style_dim * 2, 512)
            else : 
                self.fc = nn.Linear(self.bert_dim * 3, 512)

        else : 
            self.fc = nn.Linear(self.bert_dim * 4 + style_dim, 512)
            # self.fc = nn.Linear(self.bert_dim * 2 + style_dim * 2, 512)

        self.fc2 = nn.Linear(512, 1)
        # self.dropout = nn.Dropout(p=0.5)
        # self.id2emb = nn.Embedding(20, 128, padding_idx=0) 

    def get_wp_position_for_all_tokens(self, words):

        wps=[]

        # start with 1 for the inital [CLS] token
        cur=1
        for idx, word in enumerate(words):
            target=self.tokenizer.tokenize(word)
            wps.append((cur, cur+len(target)))
            cur+=len(target)

        return wps


    def get_batches(self, all_x, all_m, all_i, all_style_emb = None, quote_emb = None, batch_size=32, by_chapter=False, num_cands=10, use_all_mentions=False):

        #additional input: stylometric features of quotation text

        batches_o=[]	
        batches_x=[]
        batches_y=[]
        batches_m=[]
        batches_i = []
        # permuted_indices = np.random.permutation(len(all_x))
        for i in range(0, len(all_x), batch_size):
        # for i in range(0, 2000, batch_size):

            current_batch_input_ids=[]
            current_batch_attention_mask=[]
            current_batch_matrix_cands=[]
            current_batch_matrix_quote=[]
            current_batch_y=[]
            current_batch_eid=[]
            current_quote_eids=[]
            current_batch_style_emb = []
            current_batch_batch_eid = []
            current_batch_q_start = []
            current_batch_q_end = []
            current_batch_m_start = []
            current_batch_m_end = []
            current_batch_matrix_mentions = []
            current_batch_q_emb = []
            # batch_indices = permuted_indices[i:i+batch_size]
            # ib = [all_i[idx] for idx in batch_indices]
            # xb = [all_x[idx] for idx in batch_indices]
            # mb = [all_m[idx] for idx in batch_indices]
            ib = all_i[i:i+batch_size]
            xb = all_x[i:i+batch_size]
            mb = all_m[i:i+batch_size]
            if quote_emb is not None : 
                q_embs = quote_emb[i:i+batch_size]
            for s, sent in enumerate(xb):

                sent_wp_tokens=[self.tokenizer.convert_tokens_to_ids("[CLS]")]
                attention_mask=[1]

                for word in sent:
                    toks = self.tokenizer.tokenize(word)
                    toks = self.tokenizer.convert_tokens_to_ids(toks)
                    sent_wp_tokens.extend(toks)
                    attention_mask.extend([1]*len(toks))
                sent_wp_tokens.append(self.tokenizer.convert_tokens_to_ids("[SEP]"))
                attention_mask.append(1)

                current_batch_input_ids.append(sent_wp_tokens)
                current_batch_attention_mask.append(attention_mask)

            lengths = [len(s) for s in current_batch_input_ids]
            # Truncate to max 512 tokens
            if max(lengths) > 512: 
                pb = [i for i in range(len(lengths)) if lengths[i] > 512]
                for pb_id in pb : 
                    current_batch_input_ids[pb_id] = current_batch_input_ids[pb_id][:512]
                    current_batch_attention_mask[pb_id] = current_batch_attention_mask[pb_id][:512]

            max_len = min(512, max(lengths))

            
            for j, sent in enumerate(current_batch_input_ids):
                for k in range(len(current_batch_input_ids[j]), max_len):
                    current_batch_input_ids[j].append(0)
                    current_batch_attention_mask[j].append(0)

            if use_all_mentions : 
                max_in_batch = max([len(i[-1]) for i in mb])
                # to avoid cases where only one quote with no mentions?
                num_cands = max(max_in_batch, 1)
                
            for j, (eid, cands, quote, mentions) in enumerate(mb):
                novel_id, q_id, chapID, is_explicit = ib[j]

                wps_all=self.get_wp_position_for_all_tokens(xb[j])

                current_quote_eids.append(eid)

                if isinstance(quote, (list, tuple)) : 
                    e1_start_wp, _ = wps_all[quote[0]]
                    _, e1_end_wp = wps_all[quote[1] - 1]

                else: 
                    e1_start_wp, e1_end_wp=wps_all[quote]
                if e1_end_wp > 512: 
                    e1_end_wp = 512
                # current_batch_q_start.append(e1_start_wp)
                # current_batch_q_end.append(e1_end_wp)

                #matrix_cands=np.zeros((num_cands,max_len))
                #matrix_quote=np.zeros((num_cands,max_len))
                matrix_style_emb = []
                matrix_q_emb = []
                curr_q_s = []
                curr_q_e = []
                # if quote_emb is not None :
                #     curr_q_emb = [[]]*batch_size
                for l in range(num_cands):
                    # for k in range(e1_start_wp, e1_end_wp):
                    # 	matrix_quote[l][k]=1./(e1_end_wp-e1_start_wp) #non-zero where the QUOTE token is
                    curr_q_s.append(e1_start_wp)
                    curr_q_e.append(e1_end_wp - 1)
                    # if quote_emb is not None : 
                    #     curr_q_emb[s].append(q_embs[l])
                current_batch_q_start.append(curr_q_s)
                current_batch_q_end.append(curr_q_e)

                y=[]
                eids=[]
                batch_eids = []
                curr_m_s, curr_m_e = [], []
                if all_style_emb is not None : 
                    if not by_chapter : 
                        size = list(all_style_emb[novel_id].values())[0].size(1)
                    else : 
                        size = list(all_style_emb[novel_id].values())[0]
                        size = list(size.values())[0].size(1)
                eid2id = {}
                idcount = 1
                if use_all_mentions :
                    cands = mentions
                else :
                    # mentions are sorted by distance to quote.
                    cands = mentions[:num_cands]
                for c_idx, (start, end, truth, cand_eid) in enumerate(cands):
                    if cand_eid not in eid2id : 
                        eid2id[cand_eid] = idcount
                        idcount += 1
                    e2_start_wp, _=wps_all[start]
                    _, e2_end_wp=wps_all[end-1]
                    if all([ e2_start_wp<= 512, e2_end_wp<=512]):
                        if all_style_emb is not None : 
                            if not by_chapter: 
                                matrix_style_emb.append(all_style_emb[novel_id][cand_eid])
                            else :
                                matrix_style_emb.append(all_style_emb[novel_id][cand_eid][int(chapID)])
                        # current_batch_m_start.append(e2_start_wp)
                        # current_batch_m_end.append(e2_end_wp - 1)
                        # for k in range(e2_start_wp, e2_end_wp):
                        # 	matrix_cands[c_idx][k]=1./(e2_end_wp-e2_start_wp)  #non-zero where the candidate mention occurs
                        curr_m_s.append(e2_start_wp)
                        curr_m_e.append(e2_end_wp-1)
                        ##### ONLY CLOSEST TRUE MENTION GETS POSITIVE UPDATES
                        ##### candidates are sorted by closest distance to quote
                        y.append(truth)
                        eids.append(cand_eid)
                        batch_eids.append(eid2id[cand_eid])
                    else :
                        # candidate outside of BERT context size
                        if all_style_emb is not None : 
                            matrix_style_emb.append(torch.zeros(1,size))
                        curr_m_s.append(0)
                        curr_m_e.append(0)
                        y.append(0)
                        eids.append(None)
                        batch_eids.append(0)

                for l in range(len(y), num_cands):
                    y.append(0)
                    eids.append(None)
                    curr_m_s.append(0)
                    curr_m_e.append(0)
                    # default to all negative ones if candidate padding
                    if all_style_emb is not None : 
                        matrix_style_emb.append(torch.zeros(1,size))
                    # default to pad char ID
                    batch_eids.append(0)
                current_batch_m_start.append(curr_m_s)
                current_batch_m_end.append(curr_m_e)
                # if by_chapter: 
                # 	char_reprs = list(all_style_emb[novel_id].values())[0]
                # 	style_size = list(char_reprs.values())[0].size(1)
                # else : 
                # 	style_size = list(all_style_emb[novel_id].values())[0].size(1)
                # matrix_ids = np.zeros((max_len), dtype=np.int32)
                # eid2id = {}
                # idcount = 1
                # for m_idx, (start, end, truth, m_eid) in enumerate(mentions):
                # 	if m_eid not in eid2id : 
                # 		eid2id[m_eid] = idcount
                # 		idcount += 1
                # 	e2_start_wp, _=wps_all[start]
                # 	_, e2_end_wp=wps_all[end-1]
                # 	if all([ e2_start_wp<= 512, e2_end_wp<=512]):
                # 		matrix_ids[e2_start_wp:e2_end_wp] = eid2id[m_eid]

                # [1, 10, 512]
                if (all_style_emb is not None) : 
                    current_batch_style_emb.append(torch.cat(matrix_style_emb).unsqueeze(0))
                # if (quote_emb is not None) : 
                #     q_embs =
                #     current_batch_style_quote.append()

                #current_batch_matrix_cands.append(matrix_cands)
                #current_batch_matrix_quote.append(matrix_quote)
                current_batch_y.append(y)
                current_batch_eid.append(eids)
                current_batch_batch_eid.append(batch_eids)
                # current_batch_matrix_mentions.append(matrix_ids)
            #extra info
            batches_i.append(ib)

            batches_o.append((xb, mb)) #original
            #input x
            batches_x.append({"toks": torch.LongTensor(current_batch_input_ids), "mask":torch.LongTensor(current_batch_attention_mask)})
            #not sure what thee represent: 1/len(text)
            batch_data = {
                # "cands":torch.FloatTensor(np.array(current_batch_matrix_cands)).to(device),
                "q_start":torch.LongTensor(np.array(current_batch_q_start)),
                "q_end":torch.LongTensor(np.array(current_batch_q_end)),
                "m_start":torch.LongTensor(np.array(current_batch_m_start)),
                "m_end":torch.LongTensor(np.array(current_batch_m_end)),
                "eid": current_batch_eid,
                "cand_eids": torch.LongTensor(current_batch_batch_eid),
                # [batch_Size, 10, 512]
                # "ids" : torch.LongTensor(np.array(current_batch_matrix_mentions)),
            }
            if (all_style_emb is not None) : 
                batch_data["style_emb"] = torch.cat(current_batch_style_emb)
            if quote_emb is not None : 
                batch_data["quote_emb"] = q_embs.unsqueeze(1).repeat(1,num_cands,1)
            batches_m.append(batch_data)
            batches_y.append({"y":torch.LongTensor(current_batch_y).to(device), "eid":current_batch_eid, "quote_eids":current_quote_eids})

        return batches_x, batches_m, batches_y, batches_o, batches_i


    def forward(self, batch_x_, batch_m_): 
        batch_x = {k:v.to(device) for k,v in batch_x_.items() if isinstance(v, torch.Tensor)}
        batch_m = {k:v.to(device) for k,v in batch_m_.items() if isinstance(v, torch.Tensor)}
        batch_m["eid"] = batch_m_["eid"]

        # style scores
# 		combined=torch.cat((
# 						batch_m["quote_emb"],
# 						batch_m["style_emb"],
# 					), axis=2)
# 		style_scores = self.style_fc(combined)
# 		style_scores =self.tanh(style_scores)
# 		style_scores = self.style_fc2(style_scores)
        # cos sim score
        # style_scores = torch.bmm(batch_m["quote_emb"].unsqueeze(1), batch_m["style_emb"].transpose(2,1))
        # style_scores = style_scores.transpose(1,2)
        # spanbert scores


        _, pooled_outputs, sequence_outputs = self.bert(batch_x["toks"], token_type_ids=None, attention_mask=batch_x["mask"], output_hidden_states=True, return_dict=False)
        # input_embeddings = self.bert.embeddings.word_embeddings(batch_x["toks"])
        # input_embeddings += self.id2emb(batch_m["ids"])
        # _, pooled_outputs, sequence_outputs = self.bert(inputs_embeds=input_embeddings, token_type_ids=None, attention_mask=batch_x["mask"], output_hidden_states=True, return_dict=False)
        out=sequence_outputs[-1]

        combined_quote = []
        combined_cands = []
        # combined_style = []
        for idx, (q_s, q_e, m_s, m_e, eid) in enumerate(zip( batch_m["q_start"],  batch_m["q_end"],  batch_m["m_start"],  batch_m["m_end"], batch_m["eid"])): 
            if not self.unmask_quote : 
                combined_quote.append(out[idx, q_s])
            else : 
                combined_quote.append(torch.cat([
                    out[idx,q_s],
                    out[idx,q_e]
                ], axis = 1))
            cands = []
            styles = []
            for midd, (bm_s, bm_e, b_eid) in enumerate(zip(m_s, m_e, eid)): 
                if b_eid is None:
                    cands.append(torch.zeros_like(out[0,0].repeat(2)))
                    # styles.append(torch.zeros_like(batch_m["style_emb"][0,0]))
                else :
                    cands.append(torch.cat([
                    out[idx,bm_s],
                    out[idx,bm_e]
                ]))
                    # styles.append(batch_m["style_emb"][idx,bm_s])
            combined_cands.append(torch.stack(cands))
            # combined_style.append(torch.stack(styles))
        # combined=torch.cat((
        # 	torch.stack(combined_cands),
        # 	torch.stack(combined_quote),
        # ), axis=2)

        # if "luar" in self.mode : 
        # 	quote_repr = self.quote_fc(torch.cat(
        # 		[
        # 		torch.stack(combined_quote), batch_m["quote_emb"]
        # 	], dim = -1))
        # 	mention_repr = self.mention_fc(torch.cat(
        # 		[
        # 		torch.stack(combined_cands), batch_m["style_emb"]
        # 	], dim = -1))
        # else : 
        # 	quote_repr = torch.stack(combined_quote)
        # 	mention_repr = torch.stack(combined_cands)

        # SpanBERT repr
        if ("luar" in self.mode) or ("semantics" in self.mode)  : 
            out = torch.cat([
                torch.stack(combined_quote),
                batch_m["quote_emb"],
                torch.stack(combined_cands),
                batch_m["style_emb"],
            ], dim=-1)
        else :
            out = torch.cat([
                torch.stack(combined_quote),
                torch.stack(combined_cands),
            ], dim=-1)

        # if "luar" in self.mode : 
        # 	# Bert quote-candidate emb
        # 	# [bs, 10, H]
        # 	bert_repr =  self.bert_lin(out)
        # 	# stylistic quote-candidate emb
        # 	combined = torch.cat([
        # 	batch_m["quote_emb"],
        # 	batch_m["style_emb"]
        # ], dim=-1)
        # 	# [bs, 10, H]
        # 	style_repr = self.dropout(self.style_lin(combined))
        # 	# concatenation of both embs
        # 	# scores = bert_repr @ style_repr.transpose(2,1)
        # 	scores = torch.sum(bert_repr * style_repr, dim = 2)
        # 	return scores.unsqueeze(2)
            # out = torch.cat([bert_repr, style_repr], dim = -1)

        bert_scores = self.fc(out)
        bert_scores= self.relu(bert_scores)
        bert_scores = self.fc2(bert_scores)

        return bert_scores

    def forward2(self, batch_x, batch_m): 

        # if "luar" in self.mode : 
        # 	# BERT word embeddings
        # 	input_embeddings = self.bert.embeddings.word_embeddings(batch_x["toks"])
        # 	# Character LUAR embeddings
        # 	if "drama" not in self.mode : 
        # 		input_embeddings += self.style_lin(batch_m["style_mentions"])
        # 	else : 
        # 		# Drama LUAR has same size as spanBERT
        # 		input_embeddings += batch_m["style_mentions"]
        # 	# Input embeddings will also be added positional embeddings
        # 	_, pooled_outputs, sequence_outputs = self.bert(inputs_embeds=input_embeddings, token_type_ids=None, attention_mask=batch_x["mask"], output_hidden_states=True, return_dict=False)
        # else :
        batch_x = {k:v.to(device) for k,v in batch_x.items()}
        batch_m = {k:v.to(device) for k,v in batch_m.items()}
        _, pooled_outputs, sequence_outputs = self.bert(batch_x["toks"], token_type_ids=None, attention_mask=batch_x["mask"], output_hidden_states=True, return_dict=False)            
        out=sequence_outputs[-1]
        batch_size, _, bert_size=out.shape
        if "luar" in self.mode : 
            out += self.style_lin(batch_m["style_mentions"])

        # combined_cands=torch.matmul(batch_m["cands"],out)  #embeddings of each candidate token, mean-pooled
        # combined_quote=torch.matmul(batch_m["quote"],out) #embeddings of each quotation token, mean-pooled
        # [batch_size, bert_dim]
        combined_quote = []
        combined_cands = []
        # combined_style = []
        for idx, (q_s, q_e, m_s, m_e, eid) in enumerate(zip( batch_m["q_start"],  batch_m["q_end"],  batch_m["m_start"],  batch_m["m_end"], batch_m["eid"])): 
            if not self.unmask_quote : 
                combined_quote.append(out[idx, q_s])
            else : 
                combined_quote.append(torch.cat([
                    out[idx,q_s],
                    out[idx,q_e]
                ], axis = 1))
            cands = []
            # styles = []
            for midd, (bm_s, bm_e, b_eid) in enumerate(zip(m_s, m_e, eid)): 
                if b_eid is None:
                    cands.append(torch.zeros_like(out[0,0].repeat(2)))
                    # styles.append(torch.zeros_like(batch_m["style_mentions"][0,0]))
                else :
                    cands.append(torch.cat([
                    out[idx,bm_s],
                    out[idx,bm_e]
                ]))
                    # styles.append(batch_m["style_mentions"][idx,bm_s])
            combined_cands.append(torch.stack(cands))
            # combined_style.append(torch.stack(styles))
        preds = self.fc(combined)
        preds=self.tanh(preds)
        preds = self.fc2(preds)
        # if self.unmask_quote : 
        # 	combined_quote = out[list(range(batch_size)), batch_m["q_start"]]
        # else : 
        # 	combined_quote = torch.cat([
        # 		out[list(range(batch_size)), batch_m["q_start"]],
        # 		out[list(range(batch_size)), batch_m["q_end"]]
        # 	], axis=1)
        # combined_cands = torch.cat([
        # 	out[list(range(batch_size)), batch_m["m_start"]],
        # 	out[list(range(batch_size)), batch_m["m_end"]]
        # ], axis=1)
        # [batch_size, 10, bert_dim * 2]
        #combined_quote = combined_quote.unsqueeze(1).repeat(1,10,1)
        # char id encoding
        # char_embedding = self.char_emb(batch_m["cand_eids"])
        #append style features here
        # combined=torch.cat((combined_cands, combined_quote, char_embedding, batch_m["style_emb"]), axis=2)
        #combined=torch.cat((combined_cands, combined_quote, char_embedding), axis=2)
        if self.mode != "vanilla": 
            combined=torch.cat((
                torch.stack(combined_cands),
                torch.stack(combined_quote),
                # torch.stack(combined_style),
            ), axis=2)
        else : 
            combined=torch.cat((
                torch.stack(combined_cands),
                torch.stack(combined_quote),
            ), axis=2)
        # elif self.mode == "char_emb" :
        # 	char_embedding = self.char_emb(batch_m["cand_eids"]) 
        # 	combined=torch.cat((combined_cands, combined_quote, char_embedding), axis=2)
        # elif self.mode == "char_and_luar_emb":
        # 	char_embedding = self.char_emb(batch_m["cand_eids"])
        # 	# char_features = char_embedding + batch_m["style_emb"]
        # 	# char_features = batch_m["style_emb"]
        # 	combined=torch.cat((combined_cands, combined_quote, batch_m["style_emb"], char_embedding), axis=2)

        # elif self.mode == "luar_emb": 
        # 	# char_features = char_embedding + batch_m["style_emb"]
        # 	# char_features = batch_m["style_emb"]
        # 	combined=torch.cat((combined_cands, combined_quote, batch_m["style_emb"]), axis=2)
        preds = self.fc(combined)
        preds=self.tanh(preds)
        preds = self.fc2(preds)

        return preds



    def evaluate(self, dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, dev_i_batches, epoch):

        self.eval()

        cor=0.
        tot=0.
        nones=0
        corr_noexp = 0.
        tot_noexp = 0.
        gold_eids={}
        pred_eids={}
        gold_eids_noexp = {}
        pred_eids_noexp = {}
        with torch.no_grad():

            idd=0
            for x1, m1, y1, o1, i1 in tqdm(zip(dev_x_batches, dev_m_batches, dev_y_batches, dev_o_batches, dev_i_batches), total = len(dev_x_batches)):
                y_pred = self.forward(x1, m1)

                orig, meta=o1
                is_explicit = [int(ib[-1]) for ib in i1]
                predictions=torch.argmax(y_pred, axis=1).detach().cpu().numpy()
                for idx, pred in enumerate(predictions):

                    sent=orig[idx]
                    gold_eids[idd]=y1["quote_eids"][idx]

                    predval=y1["eid"][idx][pred[0]]
                    if predval is None:
                        predval="none-%s" % (idd)

                    pred_eids[idd]=predval
                    val=y1["y"][idx][pred[0]]

                    if pred[0] < len(meta[idx][1]):
                        ent_start, ent_end, lab, ent_eid=meta[idx][1][pred[0]]

                        # if epoch == "test":
                        # 	print("epoch %s" % epoch, ' '.join(sent[:ent_start]), PINK, ' '.join(sent[ent_start:ent_end]), "(%s)" % int(val.detach().cpu().numpy()), ENDC, ' '.join(sent[ent_end:]))

                    if val == 1:
                        cor+=1

                    if is_explicit[idx] == 0 : 
                        gold_eids_noexp[idd]=y1["quote_eids"][idx]
                        pred_eids_noexp[idd]=predval
                        if val == 1 : 
                            corr_noexp += 1 
                        tot_noexp +=1
                    tot+=1
                    idd+=1

        precision, recall, F=b3(gold_eids, pred_eids)

        print("Nones: %s" % nones)
        print("Epoch %s, [ALL] F1: %.3f\tP: %.3f, R: %.3f" % (epoch, F, precision, recall))
        print("Epoch %s, [ALL] accuracy: %.3f" % (epoch, cor/tot) )
        if len(gold_eids_noexp) >0 : 
            precision_noexp, recall_noexp, F_noexp=b3(gold_eids_noexp, pred_eids)
            acc_noexp = corr_noexp/tot_noexp
            print("Epoch %s, [NOEXP] F1: %.3f\tP: %.3f, R: %.3f" % (epoch, F_noexp, precision_noexp, recall_noexp))
            print("Epoch %s, [NOEXP] accuracy: %.3f" % (epoch, corr_noexp/tot_noexp) )
        else:
            print("Epoch %s, Found no non-explicit quotes in dev")
            acc_noexp = None
            F_noexp = None
        return F, cor/tot, F_noexp, acc_noexp





