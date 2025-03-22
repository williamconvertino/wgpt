import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments
from datasets import load_dataset
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

    if not os.path.exists('/raw'):
        os.makedirs('/raw', exist_ok=True)
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=split)
        dataset.save_to_disk('/raw')
        del dataset    

    for split in ["train", "validation", "test"]:
            
        dataset = load_dataset("/raw/DKYoon___slim_pajama-6_b", split=split, streaming=True)
        
        DiskDataset.generate_bin("slim-pajama", split, dataset['text'], tokenizer)

if __name__ == "__main__":
    process_dataset()