import random
from torch.utils.data import Sampler

class PDBSampler(Sampler):
    """
    Proportional Distributed Batch Sampler (PDBSampler)

    This sampler is designed for multi-dataset training where every batch contains exclusively samples
    from a single dataset. The dataset for each batch is chosen either:
      - With a probability proportional to its size (proportional_sampling=True), or
      - In a regular distributed manner (proportional_sampling=False).
      
    It is compatible with PyTorch Lightning's Distributed Data Parallel (DDP) training.
    """
    def __init__(self, datasets, batch_size, world_size, rank, shuffle=True, proportional_sampling=True):
        """
        Args:
            datasets (list): List of dataset objects.
            batch_size (int): Batch size.
            world_size (int): Number of distributed processes.
            rank (int): Rank of the current process.
            shuffle (bool, optional): Whether to shuffle the data. Default is True.
            proportional_sampling (bool, optional): Whether to choose batches proportionally 
                                                    based on dataset sizes. Default is True.
        """
        self.datasets = datasets
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.proportional_sampling = proportional_sampling
        
        self.batches = self._create_batches()
        self._pad_batches()
        self.batches = self.batches[self.rank::self.world_size]

    def _create_batches(self):
        batches = []
        if self.proportional_sampling:
            # Proportional sampling:
            for d_idx, dataset in enumerate(self.datasets):
                indices = list(range(len(dataset)))
                if self.shuffle:
                    random.shuffle(indices)
                num_batches = len(indices) // self.batch_size
                for i in range(num_batches):
                    batch_indices = indices[i * self.batch_size : (i + 1) * self.batch_size]
                    batches.append((d_idx, batch_indices))
            if self.shuffle:
                random.shuffle(batches)
        else:
            # Regular distributed sampling:
            all_indices = []
            for d_idx, dataset in enumerate(self.datasets):
                for idx in range(len(dataset)):
                    all_indices.append((d_idx, idx))
            if self.shuffle:
                random.shuffle(all_indices)
            batches = []
            current_batch = []
            current_dataset = None
            for d_idx, idx in all_indices:
                if current_dataset is None:
                    current_dataset = d_idx
                if d_idx == current_dataset:
                    current_batch.append(idx)
                    if len(current_batch) == self.batch_size:
                        batches.append((current_dataset, current_batch))
                        current_batch = []
                        current_dataset = None
                else:
                    current_batch = [idx]
                    current_dataset = d_idx
                    if len(current_batch) == self.batch_size:
                        batches.append((current_dataset, current_batch))
                        current_batch = []
                        current_dataset = None
        return batches

    def _pad_batches(self):
        """
        Pads the batch list so that the total number of batches is divisible by world_size.
        """
        total_batches = len(self.batches)
        remainder = total_batches % self.world_size
        if remainder != 0:
            pad_size = self.world_size - remainder
            # Pad by repeating batches from the beginning.
            self.batches.extend(self.batches[:pad_size])

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        """
        Returns the batch corresponding to the given index.
        Each batch is a tuple: (dataset_id, list_of_sample_indices).
        """
        return self.batches[index]

    def __iter__(self):
        """
        Returns an iterator over the batches.
        """
        for batch in self.batches:
            yield batch
