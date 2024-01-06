import numpy as np
import torch 
import torch.nn as nn 
from light_training.trainer import Trainer
import random 
import os 
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(42)
import os


class Segmenrator(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        
        from networks.segformer import SegFormer
        self.model = SegFormer()
        # self.load_state_dict("./best_model_0.9890.pt")
        self.load_state_dict("./final_model_0.9620.pt")

    
