import os
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class DiskDataset(Dataset):
    
    def __init__(self, dataset_name, split, seq_len, stride_ratio=0.5):

        self.seq_len = seq_len
        self.stride = max(int(seq_len * stride_ratio), 1)
        
        self.file_path = os.path.join(os.path.dirname(__file__), dataset_name, "tokenized", f"{split}.bin")
        
        self.data = np.memmap(self.file_path, mode='r', dtype=np.int32)
        
        total_length = self.data.shape[0]
        self.indices = list(range(0, total_length - seq_len + 1, self.stride))
    
    def __len__(self):

        return len(self.indices)
    
    def shuffle_indices(self, seed=None):
       
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.indices)
    
    def __getitem__(self, idx):

        start_idx = self.indices[idx]
        sample = self.data[start_idx : start_idx + self.seq_len]
        
        return torch.tensor(sample, dtype=torch.long)
    
    @staticmethod
    def generate_bin(dataset_dict, dataset_name, tokenizer, use_eos=True, buffer_size=1024):
        
        for split in dataset_dict.keys():
            
            dataset = dataset_dict[split]
            dataset = dataset.map(lambda x: { 'input_ids': tokenizer.encode(x["text"], eos=use_eos)}, batched=True, remove_columns=["text"])
            
            file_size = sum(len(sample) for sample in dataset['input_ids'])
            
            file_dir = os.path.join(os.path.dirname(__file__), dataset_name, "tokenized")
            file_path = os.path.join(file_dir, f"{split}.bin")
            
            os.makedirs(file_dir, exist_ok=True)
            
            memmap_array = np.memmap(file_path, dtype="int32", mode="w+", shape=(file_size,))
            
            buffer = []
            write_pointer = 0
            
            for sequence in tqdm(dataset["input_ids"], desc="Generating dataset files"):
                buffer.extend(sequence)
                if len(buffer) >= buffer_size:
                    memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
                    write_pointer += len(buffer)
                    buffer = []
            
            if len(buffer) > 0:
                memmap_array[write_pointer: write_pointer + len(buffer)] = buffer
                write_pointer += len(buffer)
                buffer = []
                
            memmap_array.flush()
            return memmap_array
    
    @staticmethod
    def get_splits(dataset_name, seq_len):
        
        return {
            'train': DiskDataset(dataset_name, 'train', seq_len),
            'validation': DiskDataset(dataset_name, 'validation', seq_len),
            'test': DiskDataset(dataset_name, 'test', seq_len)
        }