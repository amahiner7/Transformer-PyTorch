import os
from datetime import datetime


class FilePath(object):
    LOG_ROOT_DIR = "./log"
    TRAINING_LOG_DIR = os.path.join(LOG_ROOT_DIR, "training")
    DATETIME_DIR = datetime.now().strftime("%Y%m%d-%H%M%S")
    TENSORBOARD_LOG_DIR = os.path.join(TRAINING_LOG_DIR, DATETIME_DIR)
    MODEL_FILE_ROOT_DIR = "./model_files"
    MODEL_FILE_DIR = os.path.join(MODEL_FILE_ROOT_DIR, DATETIME_DIR)
    MODEL_FILE_NAME = "Transformer_PyTorch_epoch_{}_val_loss_{:.4f}.pth"


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory: {} is created.".format(dir_name))


def make_directories():
    make_dir(FilePath.LOG_ROOT_DIR)
    make_dir(FilePath.TRAINING_LOG_DIR)
    make_dir(FilePath.MODEL_FILE_ROOT_DIR)
