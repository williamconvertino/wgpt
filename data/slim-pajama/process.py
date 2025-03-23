import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import util.cache # Initializes cache in the data directory, to avoid home directory issues on cloud environments
import argparse
from datasets import load_dataset, DownloadConfig, get_dataset_config_info
from data.diskdataset import DiskDataset
from data.tokenizer import Tokenizer

def process_dataset(buffer_size):
    
    tokenizer = Tokenizer()
        
    dataset_dict = load_dataset("DKYoon/SlimPajama-6B", cache_dir='./raw')
    
    DiskDataset.generate_bin(dataset_dict, 'slim-pajama', tokenizer, use_eos=True, buffer_size=buffer_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=1024)
    args = parser.parse_args()
    buffer_size = args.buffer_size
    process_dataset(buffer_size)