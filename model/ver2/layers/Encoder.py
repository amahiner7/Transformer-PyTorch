from torch import nn
from model.ver2.embeddings.TransformerEmbedding import TransformerEmbedding
from model.ver2.layers.EncoderBlock import EncoderBlock


class Encoder(nn.Module):
    def __init__(self,
                 d_input, d_embed, d_model, d_ff, num_layers, num_heads, seq_len,
                 dropout_prob, device):
        super().__init__()

        self.transformer_embedding = TransformerEmbedding(vocab_size=d_input,
                                                          d_model=d_model,
                                                          seq_len=seq_len,
                                                          dropout_prob=dropout_prob,
                                                          device=device)
        self.block_list = nn.ModuleList(
            [EncoderBlock(d_embed=d_embed,
                          d_model=d_model,
                          d_ff=d_ff,
                          num_heads=num_heads,
                          dropout_prob=dropout_prob,
                          device=device)
             for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, source, mask):
        output = self.dropout(self.transformer_embedding(source))

        for block in self.block_list:
            output = block(source=output, mask=mask)

        return output
