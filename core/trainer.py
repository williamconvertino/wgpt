import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from core.lightning import DatasetLightningWrapper, ModelLightningWrapper
from data.diskdataset import DiskDataset
from util.gpu_utils import get_best_devices

# Training Constants
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
MAX_EPOCHS = 10
PRECISION = 16
MAX_DEVICES = 2
LOG_STEPS = 50
CHECKPOINT_SAVE_STEPS = 500

class Trainer:
    def __init__(self, model, tokenizer):
        """
        Initializes training by generating splits, wrapping the model,
        selecting devices, and preparing checkpoint callbacks.
        
        Args:
            model: The unwrapped model instance.
            tokenizer: The tokenizer (passed for compatibility, e.g., for further use).
        """
        
        datasets = model.config.get("datasets", [])
        if not datasets:
            raise ValueError("No datasets specified in model configuration.")
        
        seq_len = model.config.get("max_seq_len", 512)
        
        splits = DiskDataset.get_splits(model.config.dataset, seq_len)
        
        self.dataset_wrapper = DatasetLightningWrapper(splits, batch_size=BATCH_SIZE)
        self.model_wrapper = ModelLightningWrapper(model, learning_rate=LEARNING_RATE)
        
        self.device_ids = get_best_devices(model, max_gpus=MAX_DEVICES, min_vram=2.0)
        
        model_name = self.model_wrapper.model_name
        checkpoint_dir = os.path.join("checkpoints", model_name)
        recent_ckpt_path = os.path.join(checkpoint_dir, "recent.pth")
        self.resume_ckpt = recent_ckpt_path if os.path.exists(recent_ckpt_path) else None
        
        self.recent_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="recent",
            every_n_train_steps=CHECKPOINT_SAVE_STEPS,
            save_top_k=-1,
            verbose=True,
        )
        self.best_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"best-{self.model_wrapper.start_datetime}",
            monitor="val_loss",
            mode="min",
            every_n_train_steps=CHECKPOINT_SAVE_STEPS,
            save_top_k=1,
            verbose=True,
        )
    
    def train(self):
        """
        Trains the model using PyTorch Lightning with DDP (if multiple GPUs are available)
        and resumes from a checkpoint if one exists.
        """
        pl_trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            precision=PRECISION,
            accelerator="gpu" if self.device_ids else "cpu",
            devices=self.device_ids if self.device_ids else None,
            strategy="ddp" if self.device_ids and len(self.device_ids) > 1 else None,
            callbacks=[self.recent_checkpoint_callback, self.best_checkpoint_callback],
            log_every_n_steps=LOG_STEPS,
            resume_from_checkpoint=self.resume_ckpt,
        )
        
        pl_trainer.fit(self.model_wrapper, datamodule=self.dataset_wrapper)
