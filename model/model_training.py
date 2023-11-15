from .vgg_model import VGG16Model
from .prepare_dataset import dataset_generator
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def model_training(train_params):
    
    vggModel = VGG16Model(**train_params["model_params"])
    train_set, val_set = dataset_generator(**train_params["data_params"])
    vggModel.train_model(train_set, val_set)


if __name__ == "__main__":
    model_training()