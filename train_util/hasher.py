import hashlib
import torch
from collections import OrderedDict

def hash_state_dict(state_dict: OrderedDict) -> str:
    """
    Creates a stable SHA256 hash of a PyTorch state_dict.
    
    This function moves all tensors to the CPU, converts them to
    raw bytes, and hashes them along with their keys in the
    order they appear in the OrderedDict.
    """
    hasher = hashlib.sha256()
    
    # We must iterate over the OrderedDict to ensure key order
    for key, tensor in state_dict.items():
        # 1. Hash the parameter name (key)
        hasher.update(key.encode('utf-8'))
        
        # 2. Hash the tensor data
        #    - Move to CPU to be device-agnostic
        #    - Get raw bytes to be dtype-agnostic (but dtype-aware)
        tensor_bytes = tensor.cpu().numpy().tobytes()
        hasher.update(tensor_bytes)
        
    return hasher.hexdigest()