import os
import numpy as np
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class DiskDataset(Dataset):
    def __init__(self, dataset_name, split, seq_len, stride_ratio=0.5):
        """
        Initializes a DiskDataset instance.

        Args:
            dataset_name (str): Name of the dataset (e.g., "slim-pajama").
            split (str): Data split to load ("train", "validation", or "test").
            seq_len (int): Sequence length for each sample.
            stride_ratio (float, optional): Ratio used to compute the stride. 
                                            Stride = int(seq_len * stride_ratio). Default is 0.5.
        """
        
        self.seq_len = seq_len
        self.stride = max(int(seq_len * stride_ratio), 1)
        
        self.file_path = os.path.join(os.path.dirname(__file__), dataset_name, "tokenized", f"{split}.bin")
        
        self.data = np.memmap(self.file_path, mode='r', dtype=np.int32)
        
        total_length = self.data.shape[0]
        self.indices = list(range(0, total_length - seq_len + 1, self.stride))
    
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.indices)
    
    def shuffle_indices(self, seed=None):
        """
        Shuffles the sample indices. If a seed is provided, shuffling is deterministic.
        
        Args:
            seed (int, optional): Random seed for shuffling.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.indices)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample given its index. Returns a tensor of shape [seq_len].
        
        Args:
            idx (int): Index into the dataset.
        
        Returns:
            torch.Tensor: A tensor containing a sequence of token ids.
        """
        start_idx = self.indices[idx]
        sample = self.data[start_idx : start_idx + self.seq_len]
        
        return torch.tensor(sample, dtype=torch.long)
    
    @staticmethod
    def generate_bin(dataset_name, split, data, tokenizer, use_eos=True, batch_size=1000, total_samples=None):
        """
        Generates a binary file for a given dataset split.
        It tokenizes the samples from the provided HuggingFace dataset
        and writes the resulting token ids to a .bin file.
        
        The .bin file is saved at: data/{dataset_name}/tokenized/{split}.bin
        
        Args:
            dataset_name (str): Name of the dataset (e.g., "slim-pajama").
            split (str): Data split ("train", "validation", or "test").
            data (iterable): An iterable over the dataset samples.
            tokenizer: An instance of the Tokenizer class.
            batch_size (int, optional): Number of samples to process at a time.
        """
        token_ids = []

        batch = []
        
        if total_samples is None:
            total_samples = sum(1 for _ in data)
        
        for i, sample in tqdm(enumerate(data), total=total_samples, desc=f"Tokenizing {split} data"):
            batch.append(sample)
            if len(batch) == batch_size:
                encoded_batch = [tokenizer.encode(text, eos=use_eos) for text in batch]
                for encoded in encoded_batch:
                    token_ids.extend(encoded)
                batch = []
        
        if batch:
            encoded_batch = [tokenizer.encode(text, eos=True) for text in batch]
            for encoded in encoded_batch:
                token_ids.extend(encoded)
        
        token_array = np.array(token_ids, dtype=np.int32)
        
        bin_dir = os.path.join(os.path.dirname(__file__), dataset_name, "tokenized")
        os.makedirs(bin_dir, exist_ok=True)
        bin_path = os.path.join(bin_dir, f"{split}.bin")
        
        memmap_array = np.memmap(bin_path, mode='w+', dtype=np.int32, shape=token_array.shape)
        memmap_array[:] = token_array[:]
        memmap_array.flush()
        
        del memmap_array

        print(f"Generated tokenized file at: {bin_path}")
    
    @staticmethod
    def get_splits(dataset_name, seq_len):
        """
        Returns a dictionary containing DiskDataset instances for each data split.

        Args:
            dataset_name (str): Name of the dataset (e.g., "slim-pajama").
            seq_len (int): The sequence length for each sample.

        Returns:
            dict: A dictionary with keys 'train', 'validation', and 'test', each mapping
                to a DiskDataset instance.
        """
        return {
            'train': DiskDataset(dataset_name, 'train', seq_len),
            'validation': DiskDataset(dataset_name, 'validation', seq_len),
            'test': DiskDataset(dataset_name, 'test', seq_len)
        }