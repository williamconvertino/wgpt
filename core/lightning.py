import os
import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class DatasetLightningWrapper(pl.LightningDataModule):
    def __init__(self, splits, batch_size=32, num_workers=4):
        """
        Args:
            splits (dict): A dictionary with keys 'train', 'validation', and 'test' mapping to DiskDataset instances.
            batch_size (int): Batch size to use in data loaders.
            num_workers (int): Number of workers for data loading.
        """
        super().__init__()
        self.splits = splits
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = self.splits["train"]
        self.val_dataset = self.splits["validation"]
        self.test_dataset = self.splits["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False  # Shuffling is handled manually.
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def on_train_epoch_start(self):
        """
        At the start of each training epoch, shuffle the train dataset indices using the current epoch as seed.
        """
        epoch_seed = self.trainer.current_epoch
        if hasattr(self.train_dataset, "shuffle_indices"):
            self.train_dataset.shuffle_indices(seed=epoch_seed)

class ModelLightningWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate):
        """
        Wraps a model to integrate with PyTorch Lightning.

        Args:
            model: The underlying model instance.
            learning_rate (float): Learning rate for optimization.
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.start_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        self.log_dir = os.path.join("logs", self.model.name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f"{self.start_datetime}.log")
        
        self.best_train_loss = None
        self.best_val_loss = None
        self.best_val_ppl = None
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Returns:
            The predicted logits from the underlying model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step and logs the train loss.
        Assumes that batch is a tensor of token IDs of shape [B, seq_len].
        """
        logits = self.forward(batch)
        
        targets = batch[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        with open(self.log_file, "a") as f:
            f.write(f"Epoch {self.current_epoch}, Batch {batch_idx}, Train Loss: {loss.item()}\n")
            
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a validation step, logs the validation loss and perplexity,
        and updates the best metrics if the validation loss improves.
        """
        logits = self.forward(batch)
        targets = batch[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        ppl = torch.exp(loss)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ppl", ppl, prog_bar=True)
        
        with open(self.log_file, "a") as f:
            f.write(f"Epoch {self.current_epoch}, Batch {batch_idx}, Val Loss: {loss.item()}, Val PPL: {ppl.item()}\n")
        
        if self.best_val_loss is None or loss.item() < self.best_val_loss:
            train_loss_metric = self.trainer.callback_metrics.get("train_loss", loss)
            self.best_train_loss = train_loss_metric.item() if hasattr(train_loss_metric, "item") else loss.item()
            self.best_val_loss = loss.item()
            self.best_val_ppl = ppl.item()
        return loss

    def on_save_checkpoint(self, checkpoint):
        """
        Saves the model's config, current epoch, global step, and best metrics into the checkpoint.
        """
        checkpoint["config"] = self.model.config
        checkpoint["epoch"] = self.current_epoch
        checkpoint["global_step"] = self.global_step
        checkpoint["best_train_loss"] = self.best_train_loss
        checkpoint["best_val_loss"] = self.best_val_loss
        checkpoint["best_val_ppl"] = self.best_val_ppl

    def configure_optimizers(self):
        """
        Configures the AdamW optimizer and a OneCycleLR scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.learning_rate, 
            total_steps=total_steps
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
