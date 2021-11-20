import torch


class HyperParameter(object):
    BATCH_SIZE = 128
    MAX_SEQ_LEN = 100
    MODEL_DIM = 256
    EMBED_DIM = 256
    NUM_LAYERS = 3
    NUM_HEADS = 8
    FF_DIM = 512
    DROPOUT_PROB = 0.1
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 30
    CLIP = 1
    SEED_VALUE = 777
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU device setting

