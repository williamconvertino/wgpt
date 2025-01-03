import os
from tokenizers import ByteLevelBPETokenizer

def generate_bpe_files(corpus_iterator, vocab_size, output_dir):
    assert not os.path.exists(output_dir), f"Failed to generate BPE files: Target directory ({output_dir}) already exists"
    bpe_tokenizer = ByteLevelBPETokenizer() # Tiktoken doesn't have a generation method, so we have to use the tokenizers library
    bpe_tokenizer.train_from_iterator(corpus_iterator, vocab_size)
    bpe_tokenizer.save_model(output_dir)
    os.rename(os.path.join(output_dir, "vocab.json"), os.path.join(output_dir, "vocab.bpe"))