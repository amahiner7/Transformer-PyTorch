import torch
from torch import nn
from model.ver1.embeddings.PositionalEncoding import PositionalEncoding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, seq_len, dropout_prob, device, use_positional_encoding=False):
        super(TransformerEmbedding, self).__init__()

        self.device = device
        self.use_positional_encoding = use_positional_encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        if use_positional_encoding:
            self.position_embedding = PositionalEncoding(d_model, seq_len, device)
        else:
            self.position_embedding = nn.Embedding(seq_len, d_model)

        self.drop_out = nn.Dropout(p=dropout_prob)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self, x):
        if self.use_positional_encoding:
            position_embedding = self.position_embedding(x)
        else:
            batch_size = x.shape[0]
            x_len = x.shape[1]

            # position: [batch_size, source_length]
            position = torch.arange(0, x_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
            position_embedding = self.position_embedding(position)

        token_embedding = self.token_embedding(x)
        output = (token_embedding * self.scale) + position_embedding
        output = self.drop_out(output)

        return output
