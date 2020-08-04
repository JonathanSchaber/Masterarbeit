import torch

class SRL_Encoder(nn.Module):
    def __init__(self, num_labels=47, embedding_dim):
        super(SRL_Encoder, self)__init__()
        self.embeddings = nn.Embedding(num_labels, embedding_dim)
        self.encoder = nn.GRU(
                            input_size=embedding_dim,
                            hidden_size,
                            num_layers,
                            bias=True,
                            batch_first,
                            dropout,
                            bidirectional
                            )

    def forward(self):
        embeddings = self.embeddings
        output, h_n = self.encoder(embeddings)
