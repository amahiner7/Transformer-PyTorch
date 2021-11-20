import torch.nn as nn
from model.ver1.layers.MultiHeadAttention import MultiHeadAttention
from model.ver1.layers.PositionWiseFeedForward import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, d_embed, d_model, d_ff, num_heads, dropout_prob, device):
        super().__init__()

        self.self_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.self_attention_layer = MultiHeadAttention(
            d_embed=d_embed, d_model=d_model,
            num_heads=num_heads, dropout_prob=dropout_prob, device=device)

        self.feed_forward_layer = PositionWiseFeedForward(
            d_embed=d_embed, d_ff=d_ff, dropout_prob=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, source, mask):
        """
        :param source : shape (batch_size, seq_len, d_embed)
        :param mask: shape (batch_size, seq_len, seq_len)
        :return output: (batch_size, seq_len, d_embed)
        """
        # Self attention
        self_attention_output, _ = self.self_attention_layer(
            query_embed=source,
            key_embed=source,
            value_embed=source,
            mask=mask)

        # Dropout, Residual connection, Layer Norm
        self_attention_output = self.self_attention_norm(source + self.dropout(self_attention_output))

        # Position wise feed forward
        feed_forward_output = self.feed_forward_layer(self_attention_output)

        # Dropout, Residual connection, Layer Norm
        output = self.feed_forward_norm(self_attention_output + self.dropout(feed_forward_output))

        return output
