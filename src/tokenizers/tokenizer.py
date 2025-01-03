import re
import os
from tokenizer_util import generate_bpe_files
import tiktoken

NUM_RESERVED_TOKENS = 100
BASE_TOKENIZER_PATH = '../../models/tokenizers'

class Tokenzier:

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # Taken from Llama 3, helps maintain contractions, small numbers, etc.

    def __init__(self, file_path):

        assert os.path.isfile(file_path), f'Vocab file {file_path} could not be found'

        mergeable_ranks = tiktoken.load_tiktoken_bpe(file_path)
        num_base_tokens = len(mergeable_ranks)

        special_tokens = [
            '<|start_of_text|>', # The beginning of a text sequence
            '<|end_of_text}>', # The end of a text sequence
            '<|end_of_turn|>', # The end of a turn in a conversation
            '<|start_header|>', # The beginning of a document header
            '<|end_header|>', # The end of a document header
        ]

        for i in range(NUM_RESERVED_TOKENS - len(self.special_tokens)):
            special_tokens.append(f'<|reserved_{i}|>')
        
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens) # ID special tokens after the base tokens
        }
        
        self.tokenizer = tiktoken.Encoding(
            name=None,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens
        )

        # Set special token IDs as attributes
        for token, token_id in self.special_tokens.items():
            if 'reserved_' in token:
                continue 
            token = token.replace('<|', '').replace('|>', '')
            setattr(self, f'{token}_id', token_id)

    def encode(self, text, add_bot=False, add_eot=False):
        enc = self.tokenizer.encode(text)
        if add_bot:
            enc = [self.start_of_text_id] + enc
        if add_eot:
            enc = enc + [self.end_of_text_id]
        return enc

    def decode(self, enc):
        return self.tokenizer.decode(enc)