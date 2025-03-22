import os
from datasets import load_dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from data.diskdataset import DiskDataset
from data.tokenizer import Tokenizer

def process_dataset():
    """
    Downloads, processes, tokenizes, and saves the SlimPajama dataset using streaming,
    without saving a raw copy to disk.
    
    For each split (train, validation, test), the data is streamed from Hugging Face,
    tokenized on the fly, and the token ids are saved in a binary file via generate_bin.
    """
    
    tokenizer = Tokenizer()

    for split in ["train", "validation", "test"]:
        
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=split, cache_dir="./raw")
        
        DiskDataset.generate_bin("slim-pajama", split, dataset, tokenizer)

if __name__ == "__main__":
    process_dataset()