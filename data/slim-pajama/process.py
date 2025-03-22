import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments
from datasets import load_dataset
from data.diskdataset import DiskDataset
from data.tokenizer import Tokenizer

RAW_PATH = './raw'
HF_PATH = 'DKYoon/SlimPajama-6B'

def process_dataset():
    """
    Downloads, processes, tokenizes, and saves the SlimPajama dataset using streaming,
    without saving a raw copy to disk.
    
    For each split (train, validation, test), the data is streamed from Hugging Face,
    tokenized on the fly, and the token ids are saved in a binary file via generate_bin.
    """
    
    tokenizer = Tokenizer()

    if not os.path.exists(RAW_PATH): # Save it locally first, to avoid memory issues when tokenizing
        os.makedirs(RAW_PATH, exist_ok=True)
        dataset = load_dataset(HF_PATH)
        dataset = dataset.map(lambda x: x)
        dataset.save_to_disk(RAW_PATH)
        del dataset    
    
    dataset = load_dataset(RAW_PATH, streaming=True)
        
    for split in ["train", "validation", "test"]:
            
        DiskDataset.generate_bin("slim-pajama", split, dataset[split]['text'], tokenizer)

if __name__ == "__main__":
    process_dataset()