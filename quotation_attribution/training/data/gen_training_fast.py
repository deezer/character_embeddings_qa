import sys, re
from os import listdir
from os.path import isfile, join
import copy
import numpy as np
import json

from transformers import BertTokenizer

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



tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, do_basic_tokenize=False)
tokenizer.add_tokens(["[QUOTE]", "[ALTQUOTE]"], special_tokens=True)

def read_toks(filename):

	tok_sent_idx=0
	lastSent=None
	toks=[]
	with open(filename) as file:
		file.readline()
		for line in file:
			cols=line.rstrip().split("\t")
			# 	def __init__(self, paragraph_id, sentence_id, index_within_sentence_idx, token_id, text, pos, lemma, deprel, dephead, ner, startByte):
			parID=int(cols[0])
			sentenceID=int(cols[1])
			tokenID=int(cols[2])
			text=cols[7]
			pos=cols[10]
			lemma=cols[9]
			deprel=cols[12]
			dephead=int(cols[6])
			ner=cols[11]
			startByte=int(cols[3])

			if sentenceID != lastSent:
				tok_sent_idx=0

			tok=Token(parID, sentenceID, tok_sent_idx, tokenID, text, pos, lemma, deprel, dephead, ner, startByte)

			tok_sent_idx+=1
			lastSent=sentenceID
			toks.append(tok)

	return toks


def read_quotes(folder, mappers):

	all_quotes={}

	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	for filename in onlyfiles:

		if not filename.endswith("ann"):
			continue
		idd=re.sub("\.ann$", "", filename.split("/")[-1])

		all_quotes[idd]=[]
		quotes={}
		attrib={}

		with open(join(folder, filename)) as file:
			for line in file:
				cols=line.rstrip().split("\t")
				if cols[0] == "QUOTE":
					qid=cols[1]
					start_sid=int(cols[2])
					start_wid=int(cols[3])
					end_sid=int(cols[4])
					end_wid=int(cols[5])
					text=cols[6]

					quotes[qid]=(mappers[idd][(start_sid, start_wid)], mappers[idd][(end_sid, end_wid)], text)

				elif cols[0] == "ATTRIB":
					qid=cols[1]
					eid=cols[2]

					attrib[qid]=eid

		for qid in quotes:
			eid=attrib[qid]
			start_tok, end_tok, text=quotes[qid]
			all_quotes[idd].append((start_tok, end_tok,eid))
			
	return all_quotes


def read_tokens(folder):
	mapper={}
	tokens={}

	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	for filename in onlyfiles:
		if not filename.endswith(".txt"):
			continue

		idd=re.sub("\.txt$", "", filename.split("/")[-1])
		mapper[idd]={}
		tokens[idd]=[]
		tid=0
		with open(join(folder, filename)) as file:
			for sid, line in enumerate(file):
				for wid, word in enumerate(line.rstrip().split(" ")):
					mapper[idd][sid,wid]=tid
					tid+=1
					tokens[idd].append(word)

	return mapper, tokens


def read_coref(folder, mappers):
	onlyfiles = [f for f in listdir(folder) if isfile(join(folder, f))]
	corefs={}
	for filename in onlyfiles:
		if not filename.endswith("ann"):
			continue
		idd=re.sub("\.ann$", "", filename.split("/")[-1])

		corefs[idd]={}
		mentions={}
		with open(join(folder, filename)) as file:
			for line in file:
				cols=line.rstrip().split("\t")
				if cols[0] == "COREF":
					mentions[cols[1]]=cols[2]
		# print(mentions)
		with open(join(folder, filename)) as file:
			for line in file:
				cols=line.rstrip().split("\t")
				if cols[0] == "MENTION":
					tid=cols[1]
					start_sid=int(cols[2])
					start_wid=int(cols[3])
					end_sid=int(cols[4])
					end_wid=int(cols[5])
					cat=cols[7]
					if cat != "PER":
						continue
					
					start_tok, end_tok=mappers[idd][(start_sid, start_wid)], mappers[idd][(end_sid, end_wid)]
					if tid in mentions:
						eid=mentions[tid]
						if eid not in corefs[idd]:
							corefs[idd][eid]=[]
						corefs[idd][eid].append((start_tok, end_tok))

	return corefs



def get_context(tokens, quotes, corefs, fulltoks, idd, window=50):

	# fifty tokens on either side of quotation, excluding other quotations.
	end_quotes={}
	in_quotes=np.zeros(len(tokens))
	for idx, (q_start, q_end, eid) in enumerate(quotes):
		end_quotes[q_end]=idx
		for k in range(q_start, q_end+1):
			in_quotes[k]=1

	corefs_by_start={}
	for eid in corefs:
		for coref_start, coref_end in corefs[eid]:
			if coref_start not in corefs_by_start:
				corefs_by_start[coref_start]={}
			corefs_by_start[coref_start][coref_end]=eid


	for start_tok, end_tok, q_eid in quotes:

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

				(q_start, q_end, eid)=quotes[end_quotes[i]] #speaker of the other quotations in context
	
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
			print("%s\t%s\t%s\t%s\t%s" % (idd, q_eid, index, json.dumps(labels), text))

	
def read_ids(filename):
	ids=[]
	with open(filename) as file:
		for line in file:
			idd=line.rstrip()
			ids.append(idd)
	return set(ids)


if __name__ == "__main__":

	tokensFolder=sys.argv[1]
	quoteFolder=sys.argv[2]
	corefFolder=sys.argv[3]
	fullTokensFolder=sys.argv[4]
	mappers, tokens=read_tokens(tokensFolder)
	quotes=read_quotes(quoteFolder, mappers)
	corefs=read_coref(corefFolder, mappers)

	ids=read_ids(sys.argv[5])
	
	for idd in ids:

		fulltoks=read_toks("%s/%s.txt" % (fullTokensFolder, re.sub("_brat", "", idd)))
		
		get_context(tokens[idd], quotes[idd], corefs[idd], fulltoks, idd)



