"""
Helpers for single-GPU training.
"""

import io
import os
import socket
import blobfile as bf
import torch as th

# 🚀 Ensure distributed training is completely disabled
def setup_dist(local_rank=0):
    """
    Completely disable distributed training for single-GPU.
    """
    th.cuda.set_device(local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"

    print("🚨 Distributed training disabled. Running on a single GPU.")

def dev():
    """
    Get the device to use for PyTorch computations.
    """
    return th.device("cuda" if th.cuda.is_available() else "cpu")

def load_state_dict(path, **kwargs):
    """
    Load a PyTorch model state dictionary.
    """
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)

# 🚨 Removed all distributed-related calls, including sync_params()
def _find_free_port():
    """
    Find a free port on the system.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
