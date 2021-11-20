import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_embed, d_ff, dropout_prob):
        super().__init__()

        self.first_fc_layer = nn.Linear(d_embed, d_ff)
        self.second_fc_layer = nn.Linear(d_ff, d_embed)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input):
        """
        input shape: (batch_size, seq_len, d_embed)
        """

        output = self.first_fc_layer(input)  # output shape : (batch_size, seq_len, d_ff)
        output = self.relu(output)
        output = self.dropout(output)

        # output shape : (batch_size, seq_len, d_embed)
        output = self.second_fc_layer(output)

        return output
