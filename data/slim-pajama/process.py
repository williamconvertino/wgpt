import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments
from datasets import load_dataset, DownloadConfig, get_dataset_config_info
from data.diskdataset import DiskDataset
from data.tokenizer import Tokenizer
import time

class DataGenerator:
    def __init__(self, dataset, length):
        self.dataset = dataset
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __iter__(self):
        for sample in self.dataset:
            try:
                yield sample["text"]
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue

def process_dataset():
    
    tokenizer = Tokenizer()
    
    download_config = DownloadConfig(
        max_retries=20,
        cache_dir="./hf_cache"
    )
    
    for split in ["train", "validation", "test"]:
        
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True, download_config=download_config)
        length = get_dataset_config_info("DKYoon/SlimPajama-6B").splits[split].num_examples
        generator = DataGenerator(dataset, length)
    
        print(f"Processing {split} split...")
        DiskDataset.generate_bin("slim-pajama", split, generator, tokenizer, use_eos=True, batch_size=1000)


if __name__ == '__main__':
    process_dataset()