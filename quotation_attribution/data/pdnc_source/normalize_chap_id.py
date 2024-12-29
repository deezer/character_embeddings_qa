import pandas as pd 
import os
import numpy as np 
from tqdm import tqdm 

if __name__ == "__main__" :
    for folder in tqdm(os.listdir()) :
        if os.path.isdir(folder) : 
            quote_df = pd.read_csv(
                os.path.join(folder, "quote_info.csv")
            )
            chap_info = pd.read_csv(
                os.path.join(folder, "chap_info.csv")
            )
            chap_sb = chap_info["textStartByte"]
            chap_eb = chap_info["textEndByte"]
            all_cb = []
            for csb, ceb in zip(chap_sb, chap_eb) : 
                all_cb.append(np.arange(csb, ceb))
                
            quote_scid, quote_ecid = [], []
            for _,row in quote_df.iterrows() : 
                sb, eb = eval(row['qSpan'])
                
                for cid in range(len(all_cb)) : 
                    if (sb in all_cb[cid]) & (eb in all_cb[cid]) : 
                        quote_scid.append(cid)
                        quote_ecid.append(cid)
                    elif cid < len(all_cb) - 1 : 
                        if (sb in all_cb[cid]) & (eb in all_cb[cid+1]) :
                            quote_scid.append(cid)
                            quote_ecid.append(cid+1)
                            
            assert len(quote_scid) == len(quote_df), "Not enough valid cids"
            assert len(quote_ecid) == len(quote_df), "Not enough valid cids"
            
            quote_df["startChapID"] = quote_scid
            quote_df["endChapID"] = quote_ecid
            quote_df.to_csv(os.path.join(folder, "quote_info.csv"))