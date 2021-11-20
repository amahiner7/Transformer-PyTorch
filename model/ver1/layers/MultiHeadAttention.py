import torch
from torch import nn


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
        """
        batch_size = query_embed.shape[0]

        query = self.query_layer(query_embed)  # shape: (num_batch, seq_len, d_model)
        key = self.key_layer(key_embed)  # shape: (num_batch, seq_len, d_model)
        value = self.value_layer(value_embed)  # shape: (num_batch, seq_len, d_model)

        query = query.view(batch_size, -1, self.num_heads, self.d_key)  # shape: (num_batch, seq_len, num_heads, d_key)
        key = key.view(batch_size, -1, self.num_heads, self.d_key)  # shape: (num_batch, seq_len, num_heads, d_key)
        value = value.view(batch_size, -1, self.num_heads, self.d_key)  # shape: (num_batch, seq_len, num_heads, d_key)

        query = query.permute(0, 2, 1, 3)  # shape: (num_batch, num_heads, seq_len, d_key)
        key = key.permute(0, 2, 1, 3)  # shape: (num_batch, num_heads, seq_len, d_key)
        value = value.permute(0, 2, 1, 3)  # shape: (num_batch, num_heads, seq_len, d_key)

        key_t = key.permute(0, 1, 3, 2)  # key transpose shape: (num_batch, num_heads, d_key, seq_len)
        attention_score = torch.matmul(query, key_t)  # shape: (num_batch, num_heads, seq_len, seq_len)
        attention_score = attention_score / self.d_k_scale  # scaling

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e10)

        # attention_prob shape: (num_batch, num_heads, seq_len, seq_len), Softmax probability
        attention_prob = torch.softmax(attention_score, dim=-1)

        # query_attention shape: (num_batch, num_heads, seq_len, d_key)
        query_attention = torch.matmul(self.dropout(attention_prob), value)

        # query_attention shape: (num_batch, seq_len, num_heads, d_key)
        query_attention = query_attention.permute(0, 2, 1, 3).contiguous()

        # query_attention shape: (num_batch, seq_len, d_model)
        query_attention = query_attention.view(batch_size, -1, self.d_model)

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
