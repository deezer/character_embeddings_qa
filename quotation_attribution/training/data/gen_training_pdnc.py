from email.quoprimime import quote
import os, re, sys, json, csv, string, gzip
import pandas as pd
import pickle as pkl
import numpy as np
from collections import Counter, defaultdict
import random

from transformers import BertTokenizer

from sklearn.model_selection import train_test_split

sys.path.append('/workspace/speaker-attribution-acl2023')
# import bert_data_utils
# import data_utils

NOVELS = []
for folder in os.scandir('data/pdnc_source/'):
	if os.path.isdir(folder) and folder.name[0] not in ['.', '!']:
		NOVELS.append(folder.name)
NOVELS = sorted(NOVELS)

#TBD: k-fold

def read_booknlp_csv(path):
	df = pd.read_csv(path, \
				sep='\t', quoting=3, lineterminator='\n', keep_default_na=False, na_values="")

	return df


class Token:
	def __init__(self, paragraph_id, sentence_id, index_within_sentence_idx, token_id, text, pos, lemma, deprel, dephead, ner, startByte):
		self.text=text
		self.paragraph_id=paragraph_id
		self.sentence_id=sentence_id
		self.index_within_sentence_idx=index_within_sentence_idx
		self.token_id=token_id
		self.lemma=lemma
		self.pos=pos
		self.deprel=deprel
		self.dephead=dephead
		self.ner=ner
		self.startByte=startByte
		self.endByte=startByte+len(text)
		self.inQuote=False
		self.event="O"




def read_coref(root_path, mappers, novel):
	corefs = {}
	count = 0
	mendf = pd.read_csv(os.path.join(root_path, novel, 'mentions_used.csv'), index_col=0)
	for _, row in mendf.iterrows():
		mtt = row['text']
		sb = int(row['startByte'])
		eb = int(row['endByte']) - 1
		asb, aeb = get_offset_bytes(mtt, sb, eb)
		try:
			st, et = mappers[asb], mappers[aeb]
			eid = "CHAR_" + str(row['pdncID'])
			if eid not in corefs:
				corefs[eid] = []
			corefs[eid].append((st, et))
			count += 1
		except KeyError:
			pass
	print("Read {}/{} corefs".format(count, len(mendf)))

	return corefs


def read_toks(file_path):
	tokens = []
	tokdf = read_booknlp_csv(file_path)
	for _, row in tokdf.iterrows():
		parID = row['paragraph_ID']
		sentenceID = row['sentence_ID']
		tok_within = row['token_ID_within_sentence']
		tok_ID = row['token_ID_within_document']
		tok = row['word']
		lemma = row['lemma']
		pos = row['POS_tag']
		deprel = row['dependency_relation']
		dephead = row['syntactic_head_ID']
		ner = 'UNK'
		startByte = int(row['byte_onset'])
		
		tok = Token(parID, sentenceID, tok_within, tok_ID, tok, pos, lemma, deprel, dephead, ner, startByte)
		
		tokens.append(tok)
	
	return tokens

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



def read_quotes(root_folder, mappers, novel):
	quoteinf = []

	# for novel in NOVELS:
	quotedf = pd.read_csv(os.path.join(root_folder, novel, 'quote_info.csv'))
	charInf = pkl.load(open(os.path.join(root_folder, novel, 'charInfo.dict.pkl'), 'rb'))
	for _, row in quotedf.iterrows():
		sb, eb = eval(row['qSpan'])
		qtt = row['qText']
		asb, aeb = get_offset_bytes(qtt, sb, eb)
		qType = row['qType']
		qId = row['qID']
		qChapId = row["startChapID"]
		try:
			st, et = mappers[asb], mappers[aeb-1]
			speaker = row['speaker']
			speaker_id = charInf['name2id'][speaker]
			eid = "CHAR_" + str(speaker_id)
			etype = row["speakerType"]
			adressees = eval(row["addressee"])
			quoteinf.append((qId, qChapId, st, et, eid, qType, speaker, adressees, etype))
			# Remove minor characters if asked 
			# if use_minor_char:
			# 	quoteinf.append((qId, qChapId, st, et, eid, qType, speaker, adressees))
			# elif etype != "minor":
			# 	quoteinf.append((qId, qChapId, st, et, eid, qType, speaker, adressees))
	
		except KeyError:
			print("Error!!! Novel {} qID {}".format(novel, qId))

	print("Read {}/{} quotes".format(len(quoteinf), len(quotedf)))
	return quoteinf

def read_tokens(root_folder, novel):

	# for novel in NOVELS:
	mappers = {}
	tokens = []
	tokdf = read_booknlp_csv(os.path.join(root_folder, novel, novel+'.tokens'))
	counter = 0
	for _, row in tokdf.iterrows():
		# sid = row['sentence_ID']
		# tid = row['token_ID_within_sentence']
		stb = row['byte_onset']
		edb = row['byte_offset']
		for bt in range(stb, edb):
			mappers[bt] = counter
		tok = row['word']
		# mappers[novel][sid, tid] = counter
		tokens.append(tok)
		counter += 1

	return mappers, tokens


def split_explicit(exp_quotes, other_quotes):
	#move atleast 5 quotes per character to training set
	##assume data is for a single novel
	char2all = {}

	for q in exp_quotes:
		cid = q[2]
		qid = q[1]
		if cid not in char2all:
			char2all[cid] = set()
		char2all[cid].add(qid)
	
	for q in other_quotes:
		cid = q[2]
		qid = q[1]
		if cid not in char2all:
			char2all[cid] = set()
		char2all[cid].add(qid)
	
	chars2remove = set()
	for c, q in char2all.items():
		if len(q) < 10:
			chars2remove.add(c)

	exp_quotes = [x for x in exp_quotes if x[2] not in chars2remove]
	other_quotes = [x for x in other_quotes if x[2] not in chars2remove]

	char2train = {}
	char2rest = {}
	char2move = {}
	for x in exp_quotes:
		cid = x[2]
		qid = x[1]
		if cid not in char2train:
			char2train[cid] = set()
		char2train[cid].add(qid)

	for x in other_quotes:
		cid = x[2]
		qid = x[1]
		if cid not in char2rest:
			char2rest[cid] = set()
		char2rest[cid].add(qid)
	
	for c, q in char2train.items():
		if len(q) < 5:
			#have to move some
			test_q = char2rest[c]
			to_sample = min(len(test_q), 5 - len(q))
			samples = random.sample(q, to_sample)
			char2move[c] = samples
	
	#move
	for c, q in char2move.items():
		char2train[c].update(q)
		char2rest[c].difference_update(q)
	
	exp_qids = set()
	oth_qids = set()
	for c, q in char2train.items():
		exp_qids.update(q)
	for c, q in char2rest.items():
		oth_qids.update(q)

	all_quotes = exp_quotes + other_quotes
	exp_quotes = [x for x in all_quotes if x[2] in exp_qids]
	oth_quotes = [x for x in all_quotes if x[2] in oth_qids]

	return exp_quotes, oth_quotes


def get_context_data(tokens, quotes, corefs, fulltoks, idd, train_ids, window=50, use_minor_chars=False, unmask_quotes=False, num_cands=10):
	# fifty tokens on either side of quotation, excluding other quotations.
	end_quotes={}
	in_quotes=np.zeros(len(tokens))
	for idx, (qid, qChapId, q_start, q_end, eid, _, _, _, _) in enumerate(quotes):
		end_quotes[q_end]=idx
		for k in range(q_start, q_end+1):
			in_quotes[k]=1

	corefs_by_start={}
	for eid in corefs:
		for coref_start, coref_end in corefs[eid]:
			if coref_start not in corefs_by_start:
				corefs_by_start[coref_start]={}
			corefs_by_start[coref_start][coref_end]=eid
			
	all_quotes = []

	cid_counter = defaultdict(int)

	for qid, qChapId, start_tok, end_tok, q_eid, q_type, speaker_name, addressees, etype in quotes:
		if any([use_minor_chars, (not use_minor_chars) & (etype != "minor")]) : 
			if q_type == "Explicit" : 
				is_explicit = 1 
			else : 
				is_explicit = 0 
			start=end_tok
			count=0
			while start > 0 and count < window:
				if in_quotes[start] == 0:
					count+=1
				start-=1

			count=0
			end=end_tok
			while end < len(tokens) and count < window:
				if in_quotes[end] == 0:
					count+=1
				end+=1
			
			toks=[]
			cands=[]

			lastPar=None

			inserts=np.zeros(end-start, dtype=int)

			offset=0
			quote_token_count = 0
			for i in range(start, end):

				tok=fulltoks[i]

				if tok.paragraph_id != lastPar and lastPar is not None:
					toks.append("[PAR]")
					offset+=1

				if not in_quotes[i]:
					toks.append(tokens[i])
				else:
					if unmask_quotes : 
						if (i in range(start_tok, end_tok+1)) : 
							if i == start_tok:
								toks.append("[QUOTE_START]")
								offset+=1
							# if (quote_token_count < max_quote_length) : 
							# 	toks.append(tokens[i])
							# 	quote_token_count += 1
							if (end - i) > window :
								toks.append(tokens[i])
							if i == end_tok:
								toks.append("[QUOTE_END]")

								offset+=1
						else:
							offset-=1
					else : 
						offset-=1

				if i == end_tok :
					if not unmask_quotes:
						offset+=1

						toks.append("[QUOTE]")
						# toks.append("[QUOTE]")
						# offset+=1
				elif i in end_quotes:
					toks.append("[ALTQUOTE]")
					offset+=1

					# (oqid, q_start, q_end, eid, _)=quotes[end_quotes[i]] #speaker of the other quotations in context
		
					# if q_end < end_tok:
					# 	quotepos=i+inserts[i-start-1]-start
					# 	cand = oqid
					# 	if (idd in train_ids) and (oqid in train_ids[idd]):
					# 		cand = eid
					# 	cands.append((min(abs(q_end-start_tok), abs(q_start-end_tok)), quotepos, quotepos, cand, "QUOTE"))

				inserts[i-start]=offset


				lastPar=tok.paragraph_id


			for coref_start in range(start, end):

				if coref_start in corefs_by_start:
					if in_quotes[coref_start] == 0:
						for coref_end in corefs_by_start[coref_start]:
							coref_eid=corefs_by_start[coref_start][coref_end]
							if coref_end+inserts[coref_start-start]-start < len(toks):
								cands.append((min(abs(coref_end-start_tok), abs(coref_start-end_tok)), coref_start+inserts[coref_start-start]-start, coref_end+inserts[coref_start-start]-start, coref_eid, "ENT"))

			text=' '.join(toks)
			# wp_toks=tokenizer.tokenize(text)
			# if len(wp_toks) > 510:
			# 	print("too big!", len(wp_toks))
			# 	sys.exit(1)

			labels=[]
			all_mentions = []
			if len(cands) > 0:
				cands=sorted(cands)
				for cidx, (dist, s, e, eid, cat) in enumerate(cands):#[:10]:
					if (int(s)<512) & (int(e+1)<512) :
						# offset by 2 because we remove QUOTE START and QUOTE END
						if unmask_quotes: 
							if int(s) < toks.index("[QUOTE_START]") : 
								all_mentions.append((int(s),int(e+1),int(q_eid==eid), eid))
								if cidx < num_cands : 
									labels.append((int(s),int(e+1),int(q_eid==eid), eid))
							else:
								all_mentions.append((int(s-2),int(e-1),int(q_eid==eid), eid))
								if cidx < num_cands :
									labels.append((int(s-2),int(e-1),int(q_eid==eid), eid))
						else:
							all_mentions.append((int(s),int(e+1),int(q_eid==eid), eid))
							if cidx< num_cands : 
								labels.append((int(s),int(e+1),int(q_eid==eid), eid))
			# we append all quotes even if no candidates
			if not unmask_quotes :
				index=toks.index("[QUOTE]")
			else :
				# index = (int(start_tok + inserts[start_tok-start] - start), int(end_tok + inserts[start_tok-start] - start)+1)
				index = (toks.index("[QUOTE_START]")-1, toks.index("[QUOTE_END]"))
				qs_i, qs_e = toks.index("[QUOTE_START]"), toks.index("[QUOTE_END]")
				del(toks[qs_i])
				del(toks[qs_e-1])
				text=' '.join(toks)
			all_quotes.append([idd, qid, qChapId, q_eid, index, labels, text, is_explicit, speaker_name, addressees, all_mentions])
			cid_counter[q_eid] += 1

	return all_quotes


def get_context(tokens, quotes, corefs, fulltoks, idd, mode='explicit', window=50):
	# fifty tokens on either side of quotation, excluding other quotations.
	end_quotes={}
	in_quotes=np.zeros(len(tokens))
	for idx, (qid, q_start, q_end, eid, _) in enumerate(quotes):
		end_quotes[q_end]=idx
		for k in range(q_start, q_end+1):
			in_quotes[k]=1

	corefs_by_start={}
	for eid in corefs:
		for coref_start, coref_end in corefs[eid]:
			if coref_start not in corefs_by_start:
				corefs_by_start[coref_start]={}
			corefs_by_start[coref_start][coref_end]=eid
			
	all_quotes, exp_quotes, other_quotes = [], [], []

	cid_counter = defaultdict(int)

	for qid, start_tok, end_tok, q_eid, q_type in quotes:

		start=end_tok
		count=0
		while start > 0 and count < window:
			if in_quotes[start] == 0:
				count+=1
			start-=1

		count=0
		end=end_tok
		while end < len(tokens) and count < window:
			if in_quotes[end] == 0:
				count+=1
			end+=1

		toks=[]
		cands=[]

		lastPar=None

		inserts=np.zeros(end-start, dtype=int)

		offset=0
		for i in range(start, end):

			tok=fulltoks[i]

			if tok.paragraph_id != lastPar and lastPar is not None:
				toks.append("[PAR]")
				offset+=1


			if not in_quotes[i]:
				toks.append(tokens[i])
			else:
				offset-=1


			if i == end_tok:
				toks.append("[QUOTE]")
				offset+=1
			elif i in end_quotes:
				toks.append("[ALTQUOTE]")
				offset+=1

				(oqid, q_start, q_end, eid, _)=quotes[end_quotes[i]] #speaker of the other quotations in context
	
				if q_end < end_tok:
					quotepos=i+inserts[i-start-1]-start
					cands.append((min(abs(q_end-start_tok), abs(q_start-end_tok)), quotepos, quotepos, eid, "QUOTE"))

			inserts[i-start]=offset


			lastPar=tok.paragraph_id


		for coref_start in range(start, end):

			if coref_start in corefs_by_start:
				if in_quotes[coref_start] == 0:
					for coref_end in corefs_by_start[coref_start]:
						coref_eid=corefs_by_start[coref_start][coref_end]
						if coref_end+inserts[coref_start-start]-start < len(toks):
							cands.append((min(abs(coref_end-start_tok), abs(coref_start-end_tok)), coref_start+inserts[coref_start-start]-start, coref_end+inserts[coref_start-start]-start, coref_eid, "ENT"))

		text=' '.join(toks)
		wp_toks=tokenizer.tokenize(text)
		if len(wp_toks) > 500:
			print("too big!", len(wp_toks))
			sys.exit(1)

		labels=[]
		if len(cands) > 0:
			cands=sorted(cands)
			for dist, s, e, eid, cat in cands[:10]:
				labels.append((int(s),int(e+1),int(q_eid==eid), eid))
			index=toks.index("[QUOTE]")

			if mode == 'explicit':
				if q_type == 'Explicit':
					exp_quotes.append([idd, qid, q_eid, index, labels, text])
				else:
					other_quotes.append([idd, qid, q_eid, index, labels, text])

			else:
				all_quotes.append([idd, qid, q_eid, index, labels, text])
			# print("%s\t%s\t%s\t%s\t%s" % (idd, q_eid, index, json.dumps(labels), text))

		cid_counter[q_eid] += 1

	#filter minor speakers <= 10
	
	if mode == 'random':
		all_quotes = [x for x in all_quotes if cid_counter[x[2]]>=10]
		train_quotes, test_quotes = train_test_split(all_quotes, test_size=0.2, stratify = [x[2] for x in all_quotes])
		train_quotes, dev_quotes = train_test_split(train_quotes, test_size=0.15, stratify = [x[2] for x in train_quotes])
	
	elif mode == 'full':
		train_quotes = all_quotes
		dev_quotes, test_quotes = [], []
	
	print("Train/dev/test size: {}/{}/{}".format(len(train_quotes), len(dev_quotes), len(test_quotes)))
	
	return train_quotes, dev_quotes, test_quotes

def write_loo_data(writeFolder, tokensFolder, quoteFolder, corefFolder):

	os.makedirs(writeFolder, exist_ok=True)
	
	for novel in NOVELS:
		test_novel = novel
		print("Test novel: {}".format(test_novel))
		os.makedirs(os.path.join(writeFolder, test_novel), exist_ok=True)
		
		other_novels = [x for x in NOVELS if x!=test_novel]
		
		mappers, tokens=read_tokens(tokensFolder, test_novel)
		quotes=read_quotes(quoteFolder, mappers, test_novel)
		corefs=read_coref(corefFolder, mappers, test_novel)

		fulltoks=read_toks(os.path.join(tokensFolder, test_novel, novel+'.tokens'))
		train_quotes, dev_quotes, test_quotes = get_context(tokens, quotes, corefs, fulltoks, test_novel, mode='full')
		
		with open(os.path.join(writeFolder, test_novel, 'quotes.test.txt'), 'a') as f:
			for idd, qid, q_eid, index, labels, text in train_quotes:
				print("%s\t%s\t%s\t%s\t%s\t%s" % (idd, qid, q_eid, index, json.dumps(labels), text), file=f)
		
		for o_novel in other_novels:
			mappers, tokens=read_tokens(tokensFolder, o_novel)
			quotes=read_quotes(quoteFolder, mappers, o_novel)
			corefs=read_coref(corefFolder, mappers, o_novel)

			fulltoks=read_toks(os.path.join(tokensFolder, o_novel, o_novel+'.tokens'))
			train_quotes, dev_quotes, test_quotes = get_context(tokens, quotes, corefs, fulltoks, o_novel, mode='random')
			train_quotes.extend(test_quotes)
			
			with open(os.path.join(writeFolder, test_novel, 'quotes.train.txt'), 'a') as f:
				for idd, qid, q_eid, index, labels, text in train_quotes:
					print("%s\t%s\t%s\t%s\t%s\t%s" % (idd, qid, q_eid, index, json.dumps(labels), text), file=f)
				
					
			with open(os.path.join(writeFolder, test_novel, 'quotes.dev.txt'), 'a') as f:
				for idd, qid, q_eid, index, labels, text in dev_quotes:
					print("%s\t%s\t%s\t%s\t%s\t%s" % (idd, qid, q_eid, index, json.dumps(labels), text), file=f)

def read_split_file(file):
	novel2qids = {}
	with open(file, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			novel = row[0]
			qid = row[1]
			if novel not in novel2qids:
				novel2qids[novel] = []
			
			novel2qids[novel].append(qid)

	return novel2qids


def read_train_splits(folder):
	train_file = os.path.join(folder, 'train.csv')
	val_file = os.path.join(folder, 'val.csv')
	test_file = os.path.join(folder, 'test.csv')

	train_ids = read_split_file(train_file)
	val_ids = read_split_file(val_file)
	test_ids = read_split_file(test_file)

	return train_ids, val_ids, test_ids

def main(readFolder, writeFolder=None, use_minor_chars=False, window_size=100, unmask_quotes=False, pass_if_no_candidates=True, num_cands = 10):#, max_quote_length=200):

	tokensFolder='booknlpen/pdnc_output'
	quoteFolder='data/pdnc_source'
	corefFolder = 'data/pdnc_source'

	train_ids, val_ids, test_ids = read_train_splits(readFolder)
	train_data = []
	val_data = []
	test_data = []

	all_novels = sorted(list(set(list(train_ids.keys()) + list(val_ids.keys()) + list(test_ids.keys()))))

	for novel in all_novels:
		mappers, tokens=read_tokens(tokensFolder, novel)
		quotes=read_quotes(quoteFolder, mappers, novel)
		corefs=read_coref(corefFolder, mappers, novel)
		fulltoks=read_toks(os.path.join(tokensFolder, novel, novel+'.tokens'))

		novel_data = get_context_data(tokens, quotes, corefs, fulltoks, novel, train_ids, window=window_size, use_minor_chars=use_minor_chars, unmask_quotes=unmask_quotes, num_cands=num_cands)#, max_quote_length=max_quote_length)
		print("Sequences: ", novel, len(novel_data))
		# train_data.extend(novel_data)
		for row in novel_data:
			qid = row[1]
			if (novel in train_ids) and (qid in train_ids[novel]):
				if pass_if_no_candidates :
					labels = row[5]
					if len(labels) > 0 :
						train_data.append(row)
				else : 
					train_data.append(row)
			elif (novel in val_ids) and (qid in val_ids[novel]):
				if pass_if_no_candidates :
					labels = row[5]
					if len(labels) > 0 :
						val_data.append(row)
				else : 
					val_data.append(row)
			elif (novel in test_ids) and (qid in test_ids[novel]):
				# for test data, we append all quotes, even if no candidates.
				test_data.append(row)
			else:
				print("Novel {}, qid {} not part of any split".format(novel, qid))
	
	print("Train/dev/test: {}/{}/{}".format(len(train_data), len(val_data), len(test_data)))
	if writeFolder is not None:
		for split, data in zip(['train', 'dev', 'test'], [train_data, val_data, test_data]):

			with open(os.path.join(writeFolder, 'quotes.'+split+'.txt'), 'w') as f:
				# writer = csv.writer(f)
				for idd, qid, qChapID, q_eid, index, labels, text, is_explicit, speaker_name, addressees, all_mentions in data:
					if not unmask_quotes:
						print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (idd, qid, qChapID, q_eid, index, json.dumps(labels), text, is_explicit, speaker_name, json.dumps(addressees), json.dumps(all_mentions)), file=f)
					else:
						print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (idd, qid, qChapID, q_eid, json.dumps(index), json.dumps(labels), text, is_explicit, speaker_name, json.dumps(addressees), json.dumps(all_mentions)), file=f)

		return
	return train_data, val_data, test_data



if __name__=='__main__':
    # mode = sys.argv[1]
    readFolder = sys.argv[1]
    writeFolder = sys.argv[2]
    unmask_quotes=False

    # tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
    # if not unmask_quotes : 
    # 	tokenizer.add_tokens(["[QUOTE]", "[ALTQUOTE]"], special_tokens=True)
    # else : 
    # 	tokenizer.add_tokens(["[QUOTE_START]","[QUOTE_END]", "[ALTQUOTE]"], special_tokens=True)

    # dont use quotes of minor characters
    use_minor_char = False
    window_size = 200
    pass_if_no_candidates = False
    num_cands = 10
    # max_quote_length=100
    for split in range(5) : 
        print(f"Generating split {split}")
        read = os.path.join(readFolder, f"split_{split}")
        write = os.path.join(writeFolder, f"split_{split}")
        os.makedirs(write, exist_ok=True)
        print("Writing to: {}".format(write))
        main(read, write, use_minor_char, window_size=window_size, unmask_quotes=unmask_quotes, pass_if_no_candidates=pass_if_no_candidates, num_cands=num_cands)#, max_quote_length=max_quote_length)
