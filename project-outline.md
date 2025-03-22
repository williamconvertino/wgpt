# WGPT

## Project Overview

This project contains a pre-training pipeline for various LLM models with the ultimate goal of experimenting with different transformer architectures. The project is built in PyTorch with PyTorch Lightning.

## Folder Structure

```
wgpt/
├── checkpoints/
│   ├── gpt2/
│   │   ├── best.pth
│   │   └── recent.pth
├── configs/
│   ├── defaults.yml
│   ├── gpt2-tiny.yml
│   ├── gpt2.yml
│   ├── llama3-tiny.yml
│   └── llama3.yml
├── core/
│   ├── __init__.py
│   ├── evaluator.py
│   └── trainer.py
├── data/
│   ├── __init__.py
│   ├── slim-pajama/
│   │   ├── raw/
│   │   ├── tokenized/
│   │   └── process.py
│   ├── diskdataset.py
│   ├── pdbsampler.py
│   ├── splits.py
│   └── tokenizer.py
├── logs/
│   ├── gpt2/
│   │   └── 3-22-2025.log
│   ├── gpt2_tiny/
│   │   └── 3-22-2025.log
├── models/
│   ├── __init__.py
│   ├── gpt.py
│   └── llama.py
├── util/
│   ├── __init__.py
│   ├── cache.py
│   ├── config_utils.py
│   ├── gpu_utils.py
│   └── lightning_utils.py
├── .env
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

## Model Configuration

The model configurations store all the information needed for initializing and training the model. The configurations are stored in `.yml` files with the following structure:

```
{
    "model": {
        "name": "default",
        "type": "gpt",
        "d_embed": 1024,
        "n_layers": 8,
        "n_heads": 16,
        "max_seq_len": 512,
        "dropout": 0.1
    }
    "datasets":  [
        "slim-pajama"
    ]
}
```

When a configuration file is missing a value, it is replaced by the values given in `defaults.yml`.

## Implementation Details

### config_utils.py

This file deals with loading config files and has the following functions:

```
def load_config(name): # Returns model config as a python object
```

Takes in the name of a config file, loads the corresponding `.yml` file, converts it to a python object, fills in any missing values with the values in `defaults.yml`, and returns the object

```
def load_model_from_config(config): # Returns an initialized model
```

Takes in a config object, initializes the corresponding model with the config, sets `model.config` to the given config, and returns the model. It should find the model class automatically from within `models/` using `importlib` (the `config.model` field should match the filename and the class name, when all are converted to lowercase and hyphens are removed).

### models/

The models are stored in the `models/` folder. New model classes are identified automatically, but make sure the class name matches the filename (when both made lowercase and hyphens are removed). A model should have the following constructor:

```
class MyModel(nn.Module): # The corresponding file should be called mymodel.py or my-model.py
    def __init__(self, config):
```

And the following forward method:

```
def forward(self, input): # Returns logits
```

NOTE: DO NOT implement any models, this structure is just for reference.
NOTE: `LightningModule` is handled separately.

### diskdataset.py

The `DiskDataset` class is designed to efficiently read a tokenized dataset from a `.bin` file via `numpy`'s `memmap` function.

It has the constructor:

```
class DiskDataset:
    def __init__(self, dataset_name, split, seq_len, stride_ratio=0.5):
```

The `.bin` file is stored in `data/{dataset_name}/tokenized/{split}.bin`. The `stride_ratio` parameter dictates the stride of the sliding window, given by `stride = seq_len * stride_ratio`. To allow for shuffling, the constructor also initializes the item indices (accounting for the stride).

`DiskDataset` also has the following methods:

```
def __len__(self): # Returns the total number of samples
```

NOTE: This takes the stride into account
NOTE: Do NOT include the last window if it's smaller than `seq_len`

```
def shuffle_indices(self, seed=None): # Shuffles the indices
```

Note: If a seed is given, the indices are shuffled according to that seed.
Note: This shuffling is not needed when using the PDBSampler

```
def __getitem__(self, idx): # Returns a tensor sample based on the current indices
```

NOTE: `LightningDataModule` is handled separately.

```
def generate_bin(dataset_name, split, data, tokenizer): # Generates the .bin file for a given dataset
```

Generates a `.bin` file by iterating over `data` (a huggingface dataset), tokenizing the samples, and saving it (in batches) to `data/{dataset_name}/tokenized/{split}.bin`.

### pdbsampler.py

The `PDBSampler` class (which means Proportional Distributed Batch Sampler) acts as a pytorch distributed sampler with additional proportional batch compatibility. It makes sure that every batch contains exclusively samples from a single dataset, with the dataset chosen at random with a probability based on its relative size (while staying PyTorch Lightning DDP compatible).

```
class PDBSampler(Sampler):
    def __init__(self, datasets, batch_size, world_size, rank, shuffle=True, proportional_sampling=True):
```

Where `proportional_sampling` determines whether to do proportional sampling (otherwise it does regular distributed sampling).

It also has the following methods:

```
def __len__(self):
```

```
def __iter__(self):
```

### process.py

Each dataset has a corresponding `process.py` script. The datasets may be large, so all reading and preprocessing is done via streaming when possible. The process scripts all have the following function:

```
process_dataset(): # Downloads, processes, tokenizes, and saves the dataset
```

Currently, we only have the SlimPajama dataset. The `process.py` script downloads a simplified version of this dataset from HuggingFace using `load_dataset("DKYoon/SlimPajama-6B")` and stores it in `data/slim-pajama/raw`. It then tokenizes and saves the train, test, and validation splits via `DiskDataset.generate_bin()`.

### splits.py

```
def get_splits(dataset_name, seq_len): # Returns a dict of DiskDatasets
```

This function returns a dictionary as follows:

```
{
    'train': DiskDataset(dataset_name, 'train', seq_len),
    'validation': DiskDataset(dataset_name, 'validation', seq_len),
    'test': DiskDataset(dataset_name, 'test', seq_len)
}
```

### tokenizer.py

The `Tokenizer` class acts as our tokenizer. It uses `tiktoken.get_encoding("r50k_base")`, has the special tokens `"<|end_of_text|>"`, `"<|begin_of_text|>"`, and `"<|pad|>"`, and uses `pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"`.

The constructor is as follows:

```
class Tokenizer:
    def __init__(self):
```

and it has the methods:

```
def __len__(self): # Returns the vocab size (special tokens included)
```

```
def encode(self, text, eos=False, bos=False): # Returns the encoded string (or list of strings)
```

If `text` is a string, it encodes the string. If it is a list of strings, it encodes all of the strings. The `eos` parameter controls whether `self.eos_token_id` is added to the end of each encoded sequence, and `bos` controls whether `self.bos_token_id` is added to the beginning of each sequence.

```
def decode(self, sequence): # Returns the decoded sequence
```

If `sequence` is a list of integers, it decodes the sequence and returns it as a string. If `sequence` is a list of list of integers, it decodes each sequence and returns a list of strings. If `sequence` is a pytorch tensor, it converts it to either a list or a list of list of integers, and then decodes it as above.

### lightning_utils.py

This file contains wrapper classes for the PyTorch Lightning workflow.

The first class is `DatasetLightningWrapper`, which has the following constructor:

```
class DatasetLightningWrapper(pl.LightningDataModule):
    def __init__(self, splits, batch_size=32, num_workers=4):
```

And the following methods:

```
def setup(self, stage=None):
```

```
def val_dataloader(self):
```

```
def test_dataloader(self):
```

```
def on_train_epoch_start(self):
```

This tracks the current epoch and shuffles the train dataset using the current epoch as a seed.

The second class is `ModelLightningWrapper`, which has the following constructor:

```
class ModelLightningWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate):
```

Which stores the `start_datetime` when the model was initialized (for use in the logs).

And the following methods:

```
def forward(self, input): # Returns the predicted logits
```

```
def training_step(self, batch, batch_idx): # Returns the loss
```

Does a training step and logs the train loss to `logs/{model_name}/{start_datetime}.log`.

```
def validation_step(self, batch, batch_idx):
```

Does a validation step and logs the validation loss and perplexity to `logs/{model_name}/{start_datetime}.log`. If the validation loss is better than the previous best, it also updates the current best model metrics (train loss, validation loss, and validation perplexity), to be saved in the checkpoint.

```
def on_save_checkpoint(self, checkpoint):
```

Saves the model's config to the current checkpoint, as well as the current epoch, step, and the current best train loss, validation loss, and perplexity.

```
def configure_optimizers(self): # Returns the optimizer and scheduler
```

This configures the AdamW optimizer and a OneCycleLR scheduler.

### gpu_utils.py

```
def get_best_devices(model, max_gpus=1, min_vram=2.0) # Returns a list containing the selected gpu ids
```

This function looks at all the available GPUs and sorts them by available VRAM, eliminating any that have less available (unused) VRAM than `min_vram`. It then selects the top `max_gpus` options and returns them.

If there aren't any GPUs available, it throws an error.

### trainer.py

The `trainer.py` file contains constants for training.

```
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
MAX_EPOCHS = 10
PRECISION = 16
MAX_DEVICES = 2
LOG_STEPS = 50
CHECKPOINT_SAVE_STEPS = 500
```

The `Trainer` class manages the training of the model. It also stores the training The constructor is as follows:

```
class Trainer:
    def __init__(self, model, tokenizer):
```

First the splits are then generated based on model.config.datasets and wrapped in the `DatasetLightningWrapper`, and the model is wrapped in the `ModelLightningWrapper`. The devices are selected using `get_best_devices()` and the sampler is initialized.

If a `recent.pth` checkpoint exists for the model, training should be resumed from this checkpoint.

```
def train(self):
```

This trains the model using PyTorch Lightning with DDP.

The PL trainer has two callbacks. The first saves the most updated version of the model to `checkpoints/{model_name}/recent.pth`. The second saves the best version of the model based on validation loss to `checkpoints/{model_name}/best-{start_datetime}.pth`. Both of these are set to `every_n_train_steps=CHECKPOINT_SAVE_STEPS`

### evaluator.py

The `Evaluator` class manages the evaluation of the model. It is not currently implemented.

### main.py

```
def main()
```

This function is called in the `__main__` guard. It uses `argparse` to either start training (`python main --train gpt2`) or evaluation (`python main --eval gpt2`), loading the model with the corresponding config name.
