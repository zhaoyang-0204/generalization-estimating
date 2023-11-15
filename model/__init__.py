from .vgg_model import VGG16Model
from .prepare_dataset import load_dataset, dataset_by_classes
from .model_training import model_training

import json
import os
def load_model_params():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_dir, "model_param.json")) as f:
        params = json.load(f)

    return params