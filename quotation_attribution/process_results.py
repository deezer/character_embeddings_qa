import pandas as pd 
import numpy as np
import pickle as pkl  
import argparse 
import os 
from collections import defaultdict 

if __name__=="__main__" : 
    parser=argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the result folder")
    
    args=parser.parse_args()

    results = defaultdict(list)
    for split in range(5) : 
        
        with open(os.path.join(args.path, f"split_{split}", "test_results.pkl"), "rb") as f: 
            d = pkl.load(f)
            for metric, val in d.items() : 
                results[metric].append(val)
    
    mean_results = {k:np.mean(v) * 100 for k,v in results.items()}
    std_results = {k:np.std(v) * 100 for k,v in results.items()}
    
    for (metric, avg), (_,std) in zip(mean_results.items(), std_results.items()) :
        print(f"{metric.upper()}: {avg:0.1f} +/- {std:0.1f}")
        
                    