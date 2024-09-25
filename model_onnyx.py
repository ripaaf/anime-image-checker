import os
import json
from functools import lru_cache
from typing import Mapping, Tuple, Optional
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from huggingface_hub import HfFileSystem
from onnxruntime import get_available_providers, get_all_providers


hfs = HfFileSystem()


# Function to load ONNX model from Hugging Face repository
@lru_cache()
def open_model_from_repo(repository, model):
    # Create a local directory to store the repository's models
    repo_dir = os.path.join(os.getcwd(), repository.replace('/', '_'))
    
    # Search for the model in the nested subdirectory
    model_dir = os.path.join(repo_dir, model, model)
    
    # If the nested folder structure doesn't exist, try without double nesting
    if not os.path.exists(model_dir):
        model_dir = os.path.join(repo_dir, model)
    
    print(f"Loading model '{model}' from local directory '{model_dir}'")
    
    # Load the ONNX model
    runtime = _open_onnx_model(os.path.join(model_dir, 'model.onnx'))
    
    # Load the labels from meta.json
    with open(os.path.join(model_dir, 'meta.json'), 'r') as f:
        labels = json.load(f)['labels']
    
    print(f"Model '{model}' loaded successfully.")
    return runtime, labels


# Function to open the ONNX model
@lru_cache()
def _open_onnx_model(ckpt: str, provider: str = None) -> InferenceSession:
    print(f"Attempting to open ONNX model from checkpoint: {ckpt}")
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Try to use the specified provider or GPU first
    provider = provider or get_onnx_provider()
    
    try:
        print(f"Trying to load model with provider: {provider}")
        session = InferenceSession(ckpt, options, [provider])
        print(f"Model loaded with provider: {provider}")
    except Exception as e:
        print(f"Failed to load model with provider {provider}: {str(e)}")
        print("Falling back to CPUExecutionProvider...")
        provider = "CPUExecutionProvider"
        options.intra_op_num_threads = os.cpu_count()
        session = InferenceSession(ckpt, options, [provider])
        print(f"Model loaded with provider: {provider}")
    
    return session


# Function to get ONNX providers
def get_onnx_provider(provider: Optional[str] = None):
    alias = {'gpu': "CUDAExecutionProvider", "trt": "TensorrtExecutionProvider"}
    
    if not provider:
        if "CUDAExecutionProvider" in get_available_providers():
            print("Using CUDAExecutionProvider")
            return "CUDAExecutionProvider"
        else:
            print("Using CPUExecutionProvider")
            return "CPUExecutionProvider"
    elif provider.lower() in alias:
        print(f"Using alias for provider: {provider.lower()}")
        return alias[provider.lower()]
    else:
        for p in get_all_providers():
            if provider.lower() == p.lower() or f'{provider}ExecutionProvider'.lower() == p.lower():
                print(f"Using specific provider: {p}")
                return p
        raise ValueError(f'Unsupported provider {provider!r} found.')
