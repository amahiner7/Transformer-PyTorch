import argparse
from data.load_data import *
from model.ver2.Transformer import Transformer
from utils.common import display_loss
from config.file_path import make_directories


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer train")
    parser.add_argument("--epochs", help="Training epochs", type=int,
                        required=False, default=HyperParameter.NUM_EPOCHS, metavar="20")
    args = parser.parse_args()
    epochs = args.epochs

    make_directories()

    return_data = load_data()

    train_iterator = return_data['train_iterator']
    valid_iterator = return_data['valid_iterator']
    source_pad_index = return_data['source_pad_index']
    target_pad_index = return_data['target_pad_index']
    encoder_vocab_size = return_data['encoder_vocab_size']
    decoder_vocab_size = return_data['decoder_vocab_size']
    sample_source = return_data['sample_source']
    sample_target = return_data['sample_target']

    model = Transformer(d_input=encoder_vocab_size,
                        d_output=decoder_vocab_size,
                        d_embed=HyperParameter.EMBED_DIM,
                        d_model=HyperParameter.MODEL_DIM,
                        d_ff=HyperParameter.FF_DIM,
                        num_heads=HyperParameter.NUM_HEADS,
                        num_layers=HyperParameter.NUM_LAYERS,
                        dropout_prob=HyperParameter.DROPOUT_PROB,
                        source_pad_index=source_pad_index,
                        target_pad_index=target_pad_index,
                        seq_len=HyperParameter.MAX_SEQ_LEN)
    model.summary(sample_source=sample_source, sample_target=sample_target)

    history = model.train_on_epoch(train_data_loader=train_iterator,
                                   valid_data_loader=valid_iterator,
                                   epochs=epochs,
                                   log_interval=50)
    display_loss(history)
