from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class Config:

    # Model parameters
    d_vocab: int
    d_context: int
    d_embed: int
    n_head: int
    n_layer: int

    # Dropout
    dropout: float = 0.1
    
    def get_extension(self):
        return f"{self.d_context}C_{self.d_embed}E_{self.n_head}H_{self.n_layer}L"

class BaseModel(nn.Module):
    def __init__(self, config: Config):
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def generate(self, x, max_new_tokens=100, eos_token=None, return_inputs=False):
        
        input_size = x.size(1)

        for _ in range(max_new_tokens):
        
            logits, _ = self(x)
            x_new = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            x = torch.cat((x, x_new), dim=1)
        
            if eos_token is not None and x_new.item() == eos_token:
                break

        if not return_inputs:
            x = x[:, input_size:]
        
        return x

    def beam_search(self, x, max_new_tokens=100, num_beams=3, eos_token=None, return_inputs=False):
        
        input_size = x.size(1)

        beams = [{'x': x, 'score': 0, 'eos': False}]  # Initial beam
        
        for _ in range(max_new_tokens):
            
            new_sequences = []
            
            for beam in beams:
            
                # If EOS is already encountered, propagate the beam without changes
                if beam['eos']:
                    new_sequences.append(beam)
                    continue
                
                # Generate beam candidates
                logits, _ = self(beam['x'])
                topk = torch.topk(logits[:, -1, :], num_beams, dim=-1)
                
                for i in range(num_beams):
                    idx_next = topk.indices[0, i].unsqueeze(0).unsqueeze(0)
                    score = topk.values[0, i].item()
                    if idx_next.item() == eos_token:
                        new_sequences.append({
                            'x': beam['x'],
                            'score': beam['score'],
                            'eos': True
                            })
                    else:    
                        new_x = torch.cat((beam['x'], idx_next), dim=1)
                        new_sequences.append({
                            'x': new_x,
                            'score': beam['score'] + score,
                            'eos': False
                        })
            
            # Select beam based on normalized score
            new_sequences.sort(key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1), reverse=True)
            beams = new_sequences[:num_beams]
            
            # Break early if all beams have encountered EOS
            if all(beam['eos'] for beam in beams):
                break
        
        most_probable_sequence = max(beams, key=lambda seq: seq['score'] / (len(seq['x'][0]) + 1))
        
        x = most_probable_sequence['x']

        if not return_inputs:
            x = x[:, input_size:]
        
        return x