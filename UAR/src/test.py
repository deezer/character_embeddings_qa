from arguments import create_argument_parser
import argparse
from torch.utils.data.dataset import Dataset
import os 
import json
import numpy as np 
from transformers import AutoTokenizer 
import torch 

class DramaDataset(Dataset):
    def __init__(
        self, 
        params: argparse.Namespace, 
        split: str, 
        num_sample_per_author: int, 
        is_queries=True
    ):
        dataset_files = {
                "train": ("train_data.json",),
                "validation": ("val_queries.json", "val_targets.json"),
                "test": ("test_queries.json", "test_targets.json"),
        }
        self.params = params

        idx = 0 if is_queries or self.params.sanity else 1
        split = "train" if self.params.sanity else split
        filename = dataset_files[split][idx]
        self.is_test = True if split != "train" else False
        
        with open(os.path.join(params.data_path, filename),"r") as f : 
            self.data = json.load(f)
        self.num_sample_per_author=num_sample_per_author
        self.tokenizer = AutoTokenizer.from_pretrained(params.model_name)
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, index) : 
        data = self.data[index]["data"]
        authors = []
        input_ids = []
        att_masks = []
        print(len(data))
        for idx, (char_id, lines) in enumerate(data.items()) : 
            assert len(lines) >= self.num_sample_per_author * self.params.episode_length
            num_docs = len(lines) 
            authors.extend([int(char_id)] * self.num_sample_per_author)
            if not self.is_test : 
                indices = np.random.permutation(num_docs)
                lines = [lines[i] for i in indices][:self.num_sample_per_author * self.params.episode_length]
            author_data = self.tokenize_text(lines)
            # [input_ids, masks]
            author_ids, author_masks = [d.reshape(self.num_sample_per_author, -1, self.params.token_max_length) for d in author_data]
            input_ids.append(author_ids) 
            att_masks.append(author_masks)
            
        data = [torch.cat(input_ids), torch.cat(att_masks)]
        return data, torch.tensor(authors)
    
    def tokenize_text(self, all_text):
        tokenized_episode = self.tokenizer(
            all_text, 
            padding=True if self.params.use_random_windows else "max_length", 
            truncation=False if self.params.use_random_windows else True, 
            max_length=None if self.params.use_random_windows else self.params.token_max_length, 
            return_tensors='pt'
        )
        tokenized_episode =  self.reformat_tokenized_inputs(tokenized_episode)
        
        return tokenized_episode

    def reformat_tokenized_inputs(self, tokenized_episode):
        """Reformats the output from HugginFace Tokenizers.
        """
        if len(tokenized_episode.keys()) == 3:
            input_ids, _, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]
        else:
            input_ids, attention_mask = tokenized_episode.values()
            data = [input_ids, attention_mask]

        return data
    
    
if __name__=="__main__": 
    params = create_argument_parser()
    d = DramaDataset(params, split="train", num_sample_per_author=2)
    data, author = d[101]
    print(author)
    print(data[0].size(), data[1].size(), author.size())
    
    d = DramaDataset(params, split="validation", num_sample_per_author=1, is_queries=True)
    data, author = d[101]
    print(author)
    print(data[0].size(), data[1].size(), author.size())

    d = DramaDataset(params, split="validation", num_sample_per_author=1, is_queries=False)
    data, author = d[101]
    print(author)
    print(data[0].size(), data[1].size(), author.size())
    
    d = DramaDataset(params, split="test", num_sample_per_author=1, is_queries=True)
    data, author = d[101]
    print(author)
    print(data[0].size(), data[1].size(), author.size())

    d = DramaDataset(params, split="test", num_sample_per_author=1, is_queries=False)
    data, author = d[101]
    print(author)
    print(data[0].size(), data[1].size(), author.size())