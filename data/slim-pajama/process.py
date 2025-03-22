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
        dataset = load_dataset("DKYoon/SlimPajama-6B", split=split, streaming=True)
        total_samples = dataset.info.splits[split].num_examples
        
        def text_generator():
            for sample in dataset:
                text = sample.get("text", "").strip()
                if text:
                    yield text
        
        DiskDataset.generate_bin("slim-pajama", split, text_generator, tokenizer, total_samples=total_samples)

if __name__ == "__main__":
    process_dataset()