import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments
from datasets import load_dataset, DownloadConfig
from data.diskdataset import DiskDataset
from data.tokenizer import Tokenizer
import time

def data_generator(dataset):
    for sample in dataset:
        try:
            yield sample["text"]
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue

def process_dataset():
    
    tokenizer = Tokenizer()
    
    download_config = DownloadConfig(
        max_retries=10,
        retry_wait=10,
        cache_dir="./hf_cache"
    )
    
    for split in ["train", "validation", "test"]:
        
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True, download_config=download_config)
    
        print(f"Processing {split} split...")
        DiskDataset.generate_bin("slim-pajama", split, data_generator(dataset), tokenizer, use_eos=True, batch_size=1000)


if __name__ == '__main__':
    process_dataset()