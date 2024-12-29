import sys 
sys.path.append("src/")
from datasets.drama_dataset import DramaDataset
from arguments import create_argument_parser
from transformers import AutoModel
from tqdm import tqdm 
from utilities import metric as M
import torch
from models.transformer import Transformer
from datasets.utils import get_dataset
from models.huggingface_transformer import Transformer
if __name__=="__main__": 
    config = {
    "embedding_dim":1024,
    "model_name" : "sentence-transformers/all-mpnet-base-v2", 
    "gradient_checkpointing": False
}
    class ModelArgument : 
        embedding_dim = 1024
        model_name = "sentence-transformers/all-mpnet-base-v2"
        gradient_checkpointing = False
        def __init__(self, config) : 
            for key,val in config.items() :
                if hasattr(self, key) :
                    setattr(self, key, val)         
    model_arg = ModelArgument(config)
    model = Transformer(model_arg)
    print(model)

    path_to_ckpt = "output/drama_luar/lightning_logs/version_42/checkpoints/epoch=19-step=940.ckpt"
    state_dict=torch.load(path_to_ckpt)
    model.load_state_dict(state_dict["state_dict"])
    print(model)
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                