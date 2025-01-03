import math
import torch
from torch import nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, config):
        super(LearnedPositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(config.d_context + 1, config.d_embed) # Leave additional space architectures that use position N+1
    
    def forward(self, d_seq, device=None):
        return self.embedding(torch.arange(d_seq, device=device))
    
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, config):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        assert config.d_embed % 2 == 0, 'Odd dimensional d_embed is not supported'

        embedding = torch.zeros(config.d_context + 1, config.d_embed)
        positions = torch.arange(0, config.d_context + 1).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, config.d_embed, 2, dtype=torch.float) *
                            -(math.log(10000.0) / config.d_embed)))
        embedding[:, 0::2] = torch.sin(positions.float() * div_term)
        embedding[:, 1::2] = torch.cos(positions.float() * div_term)

        self.register_buffer('embedding', embedding)

    def forward(self, d_seq):
        return self.embedding[:d_seq]