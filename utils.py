import torch
import random
import numpy as np

def set_random_seed(seed):
    # 1) pick your seed once
    seed = 42

    # 2) seed everything
    random.seed(seed)                       # Python stdlib RNG
    np.random.seed(seed)                    # NumPy RNG
    torch.manual_seed(seed)                 # Torch CPU RNG
    torch.cuda.manual_seed_all(seed)        # Torch CUDA RNG

    # 3) force deterministic convolution algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False