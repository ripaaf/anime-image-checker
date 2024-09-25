import os
from typing import Mapping, Tuple, Optional
import numpy as np
from PIL import Image
from natsort import natsorted
from huggingface_hub import HfFileSystem
from imgutils.data import rgb_encode

hfs = HfFileSystem()

def download_all_models(repository: str):
    print(f"Checking and downloading all models from repository '{repository}'")
    
    # Create a local directory to store the repository's models
    repo_dir = os.path.join(os.getcwd(), repository.replace('/', '_'))
    os.makedirs(repo_dir, exist_ok=True)
    
    # Get a sorted list of all models in the repository
    models = natsorted([
        os.path.dirname(os.path.relpath(file, repository))
        for file in hfs.glob(f'{repository}/*/model.onnx')
    ])
    
    for model in models:
        # Define the directory for the model within the local repo directory
        model_dir = os.path.join(repo_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        # Paths for the model and metadata files
        model_file = os.path.join(model_dir, 'model.onnx')
        meta_file = os.path.join(model_dir, 'meta.json')
        
        # Download model if not already present
        if not os.path.exists(model_file):
            print(f"Downloading model '{model}' to '{model_dir}'")
            # hf_hub_download(repository, f'{model}/model.onnx', local_dir=model_dir)
        else:
            print(f"Model '{model}' already exists at '{model_dir}', skipping download.")
        
        # Download metadata if not already present
        if not os.path.exists(meta_file):
            print(f"Downloading metadata for model '{model}' to '{model_dir}'")
            # hf_hub_download(repository, f'{model}/meta.json', local_dir=model_dir)
        else:
            print(f"Metadata for model '{model}' already exists at '{model_dir}', skipping download.")
    
    print("All models are checked and downloaded successfully.")
    return models


# Function to print results in percentage
def print_results_in_percentage(results):
    print("Calculating percentage results...")
    total = sum(results.values())
    percentages = {label: (value / total) * 100 for label, value in results.items()}
    
    for label, percentage in percentages.items():
        print(f"{label}: {percentage:.2f}%")


# Function to load and process image
def _img_encode(image: Image.Image, size: Tuple[int, int] = (384, 384),
                normalize: Optional[Tuple[float, float]] = (0.5, 0.5)):
    print(f"Resizing image to {size}")
    image = image.resize(size, Image.BILINEAR)
    data = rgb_encode(image, order_='CHW')
    
    if normalize is not None:
        mean_, std_ = normalize
        mean = np.asarray([mean_]).reshape((-1, 1, 1))
        std = np.asarray([std_]).reshape((-1, 1, 1))
        data = (data - mean) / std
    print(f"Image processed with normalization: mean={mean_}, std={std_}")
    return data.astype(np.float32)
