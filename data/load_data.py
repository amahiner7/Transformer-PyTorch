from config.hyper_parameters import *
from data.DataLoader import DataLoader
from data.Tokenizer import Tokenizer
import time


def get_data_sample(data, sample_ratio):
    data_len = len(data)
    sample_count = int(data_len * sample_ratio)

    return data[:sample_count]


def get_sample_data(data_loader):
    for batch in data_loader:
        source = batch.src
        target = batch.trg[:, :-1]
        break

    return source, target


def load_data(sample_ratio=1.0):
    start_time = time.time()
    print("Load data start.")

    tokenizer = Tokenizer()
    data_loader = DataLoader(ext=('.de', '.en'),
                             tokenize_en=tokenizer.tokenize_en,
                             tokenize_de=tokenizer.tokenize_de,
                             init_token='<sos>',
                             eos_token='<eos>')

    train_dataset, valid_dataset, test_dataset = data_loader.make_dataset()

    if sample_ratio < 1.0:
        train_dataset = get_data_sample(data=train_dataset, sample_ratio=sample_ratio)
        valid_dataset = get_data_sample(data=valid_dataset, sample_ratio=sample_ratio)
        test_dataset = get_data_sample(data=test_dataset, sample_ratio=sample_ratio)

    data_loader.build_vocab(train_data=train_dataset, min_freq=2)

    train_iterator, valid_iterator, test_iterator = data_loader.make_iter(train_dataset,
                                                                          valid_dataset,
                                                                          test_dataset,
                                                                          batch_size=HyperParameter.BATCH_SIZE,
                                                                          device=HyperParameter.DEVICE)

    source_pad_index = data_loader.source.vocab.stoi['<pad>']
    target_pad_index = data_loader.target.vocab.stoi['<pad>']
    encoder_vocab_size = len(data_loader.source.vocab)
    decoder_vocab_size = len(data_loader.target.vocab)

    sample_source, sample_target = get_sample_data(train_iterator)

    elapsed_time = time.time() - start_time
    print("Load data complete.({:.1f} sec)".format(elapsed_time))

    return_data = \
        {'train_iterator': train_iterator, 'valid_iterator': valid_iterator, 'test_iterator': test_iterator,
         'source_pad_index': source_pad_index, 'target_pad_index': target_pad_index,
         'encoder_vocab_size': encoder_vocab_size, 'decoder_vocab_size': decoder_vocab_size,
         'train_dataset': train_dataset, 'valid_dataset': valid_dataset, 'test_dataset': test_dataset,
         'data_loader': data_loader,
         'sample_source': sample_source, 'sample_target': sample_target}

    return return_data
