import torch

class SRL_Encoder(nn.Module):
    def __init__(self, num_labels, embedding_dim):
        super(SRL_Encoder, self)__init__()
        self.embeddings = nn.Embedding(num_labels, embedding_dim)
        self.encoder = nn.GRU()

    def forward(self):
        pass
