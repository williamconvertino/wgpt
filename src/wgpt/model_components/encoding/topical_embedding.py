from torch import nn

class TopicalEmbedding(nn.Module):
    def __init__(self, config):
        super(TopicalEmbedding, self).__init__()
        self.embedding = nn.Embedding(config.d_vocab, config.d_embed)

    def get_E_wte(self, x, f_k=None):
        pass

    def tie_weights(self, linear_layer):
        self.embedding.weight = linear_layer.weight

    def forward(self, x):
        return self.embedding(x)