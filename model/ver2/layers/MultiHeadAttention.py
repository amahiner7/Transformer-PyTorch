import torch
from torch import nn
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, d_embed, d_model, num_heads, dropout_prob, device):
        super().__init__()

        self.d_model = d_model  # Model dimension = d_key * num_heads
        self.d_embed = d_embed  # Embedding dimension
        self.num_heads = num_heads  # Num of heads
        self.d_key = d_model // num_heads  # Key(=Query=Value) dimension

        self.query_layer = nn.Linear(d_embed, d_model)  # Query fully connected layer
        self.key_layer = nn.Linear(d_embed, d_model)  # Key fully connected layer
        self.value_layer = nn.Linear(d_embed, d_model)  # Value fully connected layer

        self.output_layer = nn.Linear(d_model, d_embed)

        self.dropout = nn.Dropout(dropout_prob)

        self.d_k_scale = torch.sqrt(torch.FloatTensor([self.d_key])).to(device)

    def _scale_dot_product_attention(self, query_embed, key_embed, value_embed, mask=None):
        """
        :param query_embed: shape (num_batch, seq_len, d_embed)
        :param key_embed: shape (num_batch, seq_len, d_embed)
        :param value_embed: shape (num_batch, seq_len, d_embed)
        :param mask: shape (num_batch, seq_len, seq_len)
        :return query_attention: shape (num_batch, num_heads, seq_len, d_model)
                attention_prob: (num_batch, num_heads, seq_len, seq_len)
        """

        query = self.query_layer(query_embed)  # shape: (num_batch, seq_len, d_model)
        key = self.key_layer(key_embed)  # shape: (num_batch, seq_len, d_model)
        value = self.value_layer(value_embed)  # shape: (num_batch, seq_len, d_model)

        # query shape: (num_batch, num_heads, seq_len, d_key)
        query = rearrange(query, 'b n (h d) -> b h n d', h=self.num_heads)
        # key shape: (num_batch, num_heads, seq_len, d_key)
        key = rearrange(key, 'b n (h d) -> b h n d', h=self.num_heads)
        # value shape: (num_batch, num_heads, seq_len, d_key)
        value = rearrange(value, 'b n (h d) -> b h n d', h=self.num_heads)

        # attention_score shape: (num_batch, num_heads, seq_len, seq_len)
        attention_score = torch.einsum('bhqd, bhkd -> bhqk', query, key)
        attention_score = attention_score / self.d_k_scale  # scaling

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)

        # attention_prob shape: (num_batch, num_heads, seq_len, seq_len), Softmax probability
        attention_prob = torch.softmax(attention_score, dim=-1)
        attention_prob = self.dropout(attention_prob)

        # query_attention shape: (num_batch, num_heads, seq_len, d_key)
        query_attention = torch.einsum('bhal, bhlv -> bhav', attention_prob, value)

        # query_attention shape: (num_batch, seq_len, d_model)
        query_attention = rearrange(query_attention, 'b h n d -> b n (h d)')

        return query_attention, attention_prob

    def forward(self, query_embed, key_embed, value_embed, mask=None):
        # query_attention shape: (num_batch, seq_len, d_model)
        query_attention, attention_prob = self._scale_dot_product_attention(query_embed=query_embed,
                                                                            key_embed=key_embed,
                                                                            value_embed=value_embed,
                                                                            mask=mask)

        # output shape: (num_batch, seq_len, d_embed)
        output = self.output_layer(query_attention)

        return output, attention_prob
