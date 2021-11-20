import torch.nn as nn
from model.ver2.layers.MultiHeadAttention import MultiHeadAttention
from model.ver2.layers.PositionWiseFeedForward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_embed, d_model, d_ff, num_heads, dropout_prob, device):
        super().__init__()

        self.self_attention_norm = nn.LayerNorm(d_model)
        self.encoder_attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)

        self.self_attention_layer = MultiHeadAttention(
            d_embed=d_embed, d_model=d_model,
            num_heads=num_heads, dropout_prob=dropout_prob, device=device)

        self.encoder_attention_layer = MultiHeadAttention(
            d_embed=d_embed, d_model=d_model,
            num_heads=num_heads, dropout_prob=dropout_prob, device=device)

        self.feed_forward_layer = PositionWiseFeedForward(
            d_embed=d_embed, d_ff=d_ff, dropout_prob=dropout_prob)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, decoder_source, decoder_mask, encoder_source, encoder_mask):
        """
        :param decoder_source: shape (batch_size, seq_len, d_embed)
        :param decoder_mask: shape (batch_size, seq_len, seq_len)
        :param encoder_source : shape (batch_size, seq_len, d_embed)
        :param encoder_mask: shape (batch_size, seq_len, seq_len)
        :return output: (batch_size, seq_len, d_embed)
        """
        # Self attention
        self_attention_output, _ = self.self_attention_layer(
            query_embed=decoder_source,
            key_embed=decoder_source,
            value_embed=decoder_source,
            mask=decoder_mask)

        # Dropout, Residual connection, Layer Norm
        self_attention_output = self.self_attention_norm(decoder_source + self.dropout(self_attention_output))

        # Encoder's attention
        encoder_attention_output, attention_prob = self.encoder_attention_layer(
            query_embed=self_attention_output,
            key_embed=encoder_source,
            value_embed=encoder_source,
            mask=encoder_mask)

        # Dropout, Residual connection, Layer Norm
        encoder_attention_output = self.encoder_attention_norm(
            self_attention_output + self.dropout(encoder_attention_output))

        # Position wise feed forward
        feed_forward_output = self.feed_forward_layer(encoder_attention_output)

        # Dropout, Residual connection, Layer Norm
        output = self.feed_forward_norm(encoder_attention_output + self.dropout(feed_forward_output))

        return output, attention_prob
