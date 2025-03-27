import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments
import argparse
from datasets import load_dataset, concatenate_datasets
from data.diskdataset import DiskDataset
from data.tokenizer import Tokenizer

def process_dataset(buffer_size):
    
    tokenizer = Tokenizer()
        
    dataset = load_dataset("roneneldan/TinyStories", cache_dir='./raw')
    dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
            
    train_test_splits = dataset.train_test_split(test_size=10000, shuffle=True, seed=42)

    train_dataset = train_test_splits["train"]
    test_dataset = train_test_splits["test"]
    
    train_val_split = train_dataset.train_test_split(test_size=10000, shuffle=True, seed=42)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]
    
    dataset_dict = {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }
            
    DiskDataset.generate_bin(dataset_dict, 'tiny-stories', tokenizer, use_eos=True, buffer_size=buffer_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=1024)
    args = parser.parse_args()
    buffer_size = args.buffer_size
    process_dataset(buffer_size)