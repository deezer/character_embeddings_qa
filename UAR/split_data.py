import json 
import argparse
import numpy as np 
import copy
import os 

def print_metadata(acts) : 
    speakers = []
    num_quotes = 0
    plays = []
    
    for act in acts: 
        speakers.append(len(act["data"]))
        plays.append(act["play"])
        num_quotes += sum([len(i) for i in act["data"].values()])
    num_unique_plays = len(set(plays))
    num_unique_speakers = sum(speakers)
    avg_num_sp_in_act = num_unique_speakers / len(speakers)
    
    print(f"# acts: {len(acts)}\t # plays: {num_unique_plays}\t # unique character: {num_unique_speakers}\t avg # speaker in act: {avg_num_sp_in_act:0.1f}\t total # of quotes: {num_quotes}")

def separate_queries_from_target(full_data, acts, min_quotes = 5) :
    char_counter = 0 
    queries = []
    targets = []
    query_index = 0 
    play_index = {}
    play_cnt = 0 
    for act_index, act in enumerate(acts): 
        if act["play"] not in play_index: 
            play_index[act["play"]] = play_cnt 
            play_cnt += 1 
        for char_id, lines in act["data"].items() : 
            num_lines = len(lines)
            num_in_q = int(num_lines / 2)
            indices = np.random.permutation(num_lines)
            query_lines = [lines[i] for i in indices[:num_in_q]]
            target_lines = [lines[i] for i in indices[num_in_q:]]
            
            queries.append(
                {
                    "data" : { int(char_id) + char_counter : query_lines},
                    "act_index" : act_index,
                    "play_index" : play_index[act["play"]]
                })
            targets.append(
                {
                    "data" : { int(char_id) + char_counter : target_lines},
                    "act_index" : act_index,
                    "play_index" : play_index[act["play"]]
                })
            # Now we parse all lines from different character in the same play
            # target_play = full_data[act["play"]]
            # for target_act in target_play["acts"] : 
            #     for target_char_id, target_lines in target_act.items() : 
            #         if (int(target_char_id) != int(char_id)) & (len(target_lines) > min_quotes): 
            #             targets.append(
            #                 {
            #                     "data" : { int(target_char_id) + char_counter : target_lines},
            #                     "query_index" : query_index,
            #                 })
            query_index += 1
        char_counter += int(char_id) + 1
    return queries, targets, play_index

def flatten_into_acts(parsed_data) : 
    act_data = []
    for play_name, play_data in parsed_data.items(): 
        for idx, act in enumerate(play_data["acts"]) :
            tmp_act = copy.deepcopy(act)
            # Delete null entries
            for char_idx in act : 
                if len(act[char_idx]) == 0 :
                    tmp_act.pop(char_idx)
            tmp_act2 = copy.deepcopy(tmp_act)
            # Refactor character ids
            for char_idx, (old_char_id,_) in enumerate(tmp_act.items()):
                tmp_act2[char_idx] = tmp_act2.pop(old_char_id)
                    
            act_data.append({"data":tmp_act2, "play":play_name})
    return act_data

def print_target_query_metadata(queries, targets) : 
    length = []
    for q in queries: 
        query_index = q["act_index"]
        
        q_targ = [i for i  in targets if i["act_index"] ==query_index]
        length.append(len(q_targ))
    print(f"# queries: {len(queries)}\t avg # target/queries: {np.mean(length)}")

if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", help="path to source data in json format")
    parser.add_argument("--dest_folder", help="path to destination folder")
    parser.add_argument("--min_num_quotes", help="minimum number of lines per character in a segmentation unit", type=int, default=32)
    parser.add_argument("--min_target_num_quotes", help="minimum number of lines per character in a segmentation unit", type=int, default=5)
    parser.add_argument("--train_size", help="size in percentage of training plays", type=float, default=0.8)
    parser.add_argument("--val_size", help="size in percentage of validation plays", type=float, default=0.1)
    parser.add_argument("--max_char_in_segment", help="Maximum number of characters in a segment unit", type=int, default=None)

    # parser.add_argument("--test_size", help="size in percentage of validation plays", type=float, default=0.1)

    # parser.add_argument("--test_size", help="size in percentage of test plays", type=float, default=0.1)

    args = parser.parse_args()
    np.random.seed(777)
    
    with open(args.source_data, "r") as f : 
        data = json.load(f)
    
    ### Delete acts/plays that do not contain enough lines/characters ###
    ### This will be useful to create train data as well as val/test queries
    tmp = copy.deepcopy(data)
    
    indices = np.random.permutation(len(data))
    # For test data, we randomly select 10% of plays 
    # For validation, we first flatten all plays by act and randomly samples 10% of these acts
    tr_size = int(args.train_size * len(indices))
    val_size = tr_size + int(args.val_size * len(indices))
    train_indices = indices[:tr_size]
    
    for play_idx, (play_name, play_data) in enumerate(data.items()) : 
        # if play_data["metadata"]["segment_unit"] != "none" : 
        v_play = 0 
        to_remove = []
        if play_idx in train_indices : 
            #  Remove characters in training acts that have a number of lines < `min_num_quotes`
            for idx, act in enumerate(play_data["acts"]) : 
                for char_id, lines in act.items() : 
                    if len(lines) < args.min_num_quotes : 
                        tmp[play_name]["acts"][idx].pop(char_id)
        else : 
            #  Remove characters in tes/val acts that have no lines
            for idx, act in enumerate(play_data["acts"]) : 
                for char_id, lines in act.items() : 
                    if len(lines) < 2 : 
                        tmp[play_name]["acts"][idx].pop(char_id)
                        
        for idx, act in enumerate(play_data["acts"]) : 
            if len(tmp[play_name]["acts"][idx]) <= 1 : 
                to_remove.append(idx)
            if args.max_char_in_segment is not None : 
                if len(tmp[play_name]["acts"][idx]) >= args.max_char_in_segment : 
                    to_remove.append(idx)
        # Remove acts that contain 1 or less characters and more than 20 characters
        tmp[play_name]["acts"] = [tmp[play_name]["acts"][i] for i in range(len(tmp[play_name]["acts"])) if i not in to_remove]

    # Remove plays that have no data left
    # parsed_data = {k:v for k,v in tmp.items() if len(v["acts"]) > 0 }
    parsed_data = tmp
    # Split train/val/test by plays
    print(f"[TOTAL] # remaining plays: {len(parsed_data)}")
    # indices = np.random.permutation(len(parsed_data))
    # # For test data, we randomly select 10% of plays 
    # # For validation, we first flatten all plays by act and randomly samples 10% of these acts
    # tr_size = int(args.train_size * len(indices))
    # val_size = tr_size + int(args.val_size * len(indices))
    # train_data = {k:v for idx,(k,v) in enumerate(parsed_data.items()) if idx in indices[:tr_size]}
    # val_size = tr_size + int(args.val_size * len(indices))
    # train_data = flatten_into_acts(train_data)

    tr_data = {k:v for idx,(k,v) in enumerate(parsed_data.items()) if (idx in train_indices)}

    tr_data = flatten_into_acts(tr_data)
    # tr_data = [train_data[i] for i in indices[val_size:]]
    print("[TRAIN]")
    print_metadata(tr_data)
    
    if not os.path.exists(args.dest_folder) : 
        os.makedirs(args.dest_folder)
        
    with open(os.path.join(args.dest_folder, "train_data.json"), "w") as f :
        json.dump(tr_data, f)
    
    val_data = {k:v for idx,(k,v) in enumerate(parsed_data.items()) if (idx in indices[tr_size:val_size])}
    val_data = flatten_into_acts(val_data)
    # val_data = [train_data[i] for i in indices[:val_size]]
    print("[VALIDATION]")
    print_metadata(val_data)

    test_data = {k:v for idx,(k,v) in enumerate(parsed_data.items()) if (idx in indices[val_size:])}
    test_data = flatten_into_acts(test_data)
    # test_data = [parsed_data[i] for i in indices[val_size:]]
    print("[TEST]")
    print_metadata(test_data)
    
    # Validation and Test data are separated into queries and targets
    # Queries: N/2 lines from character in act X 
    # true target: other N/2 lines from same character in act X
    # other targets: lines from every other character in same play
    # --> Disjoint set of lines between queries and targets
    # Among targets, we assign the right target as the other N/2 lines in act X
    
    tr_q, tr_t, _ = separate_queries_from_target(data, tr_data, min_quotes=args.min_target_num_quotes)
    print("[TRAIN]")
    print_target_query_metadata(tr_q, tr_t)
    
    val_q, val_t, val_play_index = separate_queries_from_target(data, val_data, min_quotes=args.min_target_num_quotes)
    print("[VALIDATION]")
    print_target_query_metadata(val_q, val_t)
    with open(os.path.join(args.dest_folder, "val_queries.json"), "w") as f :
        json.dump(val_q, f)
    with open(os.path.join(args.dest_folder, "val_targets.json"), "w") as f :
        json.dump(val_t, f)
    with open(os.path.join(args.dest_folder, "val_play_index.json"), "w") as f :
        json.dump(val_play_index, f)
        
    test_q, test_t, test_play_index = separate_queries_from_target(data, test_data, min_quotes=args.min_target_num_quotes)
    print("[TEST]")
    print_target_query_metadata(test_q, test_t)
    with open(os.path.join(args.dest_folder, "test_queries.json"), "w") as f :
        json.dump(test_q, f)
    with open(os.path.join(args.dest_folder, "test_targets.json"), "w") as f :
        json.dump(test_t, f)
    with open(os.path.join(args.dest_folder, "test_play_index.json"), "w") as f :
        json.dump(test_play_index, f)
        
        
        
        
        
        