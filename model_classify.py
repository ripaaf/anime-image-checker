from typing import Mapping, Tuple, Optional
from PIL import Image
from download_models import _img_encode
from model_onnyx import open_model_from_repo
from imgutils.data import load_image
from natsort import natsorted
from repo_map import *

class Classification:
    def __init__(self, title: str, repository: str, default_model=None, imgsize: int = 384):
        self.repo_models = repo_models

        print(f"Initializing task '{title}' with repository '{repository}'")
        self.title = title
        self.repository = repository

        if self.repository in self.repo_models:
            self.models = self.repo_models[self.repository]
            print(self.models)
            print(f"Found '{self.repository}' repository.")
        else:
            self.models = []
            print(self.models)
            print(f"Repository '{self.repository}' not found.")

        self.default_model = default_model or self.models[0]
        self.imgsize = imgsize

    def _gr_classification(self, image: Image.Image, model_name: str, size=384) -> Mapping[str, float]:
        print(f"Running classification for model '{model_name}' with image size {size}")
        image = load_image(image, mode='RGB')
        input_ = _img_encode(image, size=(size, size))[None, ...]
        model, labels = open_model_from_repo(self.repository, model_name)
        print(f"Performing inference on the image...")
        output, = model.run(['output'], {'input': input_})
        print(f"Inference completed.")
        values = dict(zip(labels, map(lambda x: x.item(), output[0])))
        return values
