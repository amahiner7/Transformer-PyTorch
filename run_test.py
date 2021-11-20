import argparse
import glob
import math
from config.file_path import *
from data.load_data import *
from model.ver2.Transformer import Transformer
from utils.common import *


def load_model(model_file_path):
    if os.path.isfile(model_file_path):
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

        model.load(model_file_path)
        model.summary(sample_source=sample_source, sample_target=sample_target)
    else:
        raise Exception("It can't find model file.")

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transformer test")
    parser.add_argument("--model-file-path", help="Model file directory", type=str, required=True)

    args = parser.parse_args()
    model_file_path = args.model_file_path

    return_data = load_data()
    test_iterator = return_data['test_iterator']
    test_dataset = return_data['test_dataset']
    source_pad_index = return_data['source_pad_index']
    target_pad_index = return_data['target_pad_index']
    encoder_vocab_size = return_data['encoder_vocab_size']
    decoder_vocab_size = return_data['decoder_vocab_size']
    data_loader = return_data['data_loader']
    sample_source = return_data['sample_source']
    sample_target = return_data['sample_target']

    model = load_model(model_file_path=model_file_path)

    test_loss_list = model.evaluate(data_loader=test_iterator)
    test_loss = np.mean(test_loss_list)

    print("Test Loss: {:.4f}, PPL: {:.4f}".format(test_loss, math.exp(test_loss)))

    example_idx = 10
    src = vars(test_dataset.examples[example_idx])['src']
    trg = vars(test_dataset.examples[example_idx])['trg']
    print(f'Source sentence: {src}')
    print(f'Target sentence: {trg}')
    translation, attention = translate_sentence(src, data_loader.source, data_loader.target,
                                                model, HyperParameter.DEVICE, logging=True)
    print("Model output:", " ".join(translation))
    display_attention(src, translation, attention)
    show_bleu(test_dataset, data_loader.source, data_loader.target, model, HyperParameter.DEVICE)
