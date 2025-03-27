import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from core.lightning import DatasetLightningWrapper, ModelLightningWrapper
from data.diskdataset import DiskDataset
from util.gpu_utils import get_best_devices

# Training Constants
BATCH_SIZE = 128
LEARNING_RATE = 5e-5
MAX_EPOCHS = 10
PRECISION = "16-mixed"
LOG_STEPS = 50
CHECKPOINT_SAVE_PCT = 0.05

class Trainer:
    def __init__(self, model, tokenizer, max_gpus=2):
        seq_len = model.config.max_seq_len
        
        splits = DiskDataset.get_splits(model.config.dataset, seq_len)
        
        self.dataset_wrapper = DatasetLightningWrapper(splits, batch_size=BATCH_SIZE)
        self.model_wrapper = ModelLightningWrapper(model, learning_rate=LEARNING_RATE)
        
        self.device_ids = get_best_devices(model, max_gpus=max_gpus, min_vram=2.0)
        
        assert len(self.device_ids) > 0, "No GPUs available with at least 2GB of free VRAM."
        
        model_name = self.model_wrapper.name
        checkpoint_dir = os.path.join("checkpoints", model_name)
        recent_ckpt_path = os.path.join(checkpoint_dir, "recent.pth")
        self.resume_ckpt = recent_ckpt_path if os.path.exists(recent_ckpt_path) else None
        
        self.checpoint_save_steps = int(len(splits["train"]) * CHECKPOINT_SAVE_PCT)
        
        self.recent_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="recent",
            every_n_train_steps=self.checpoint_save_steps,
            save_top_k=-1,
            verbose=True,
        )
        self.best_checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"best-{self.model_wrapper.start_datetime}",
            monitor="val_loss",
            mode="min",
            every_n_train_steps=self.checpoint_save_steps,
            save_top_k=1,
            verbose=True,
        )
    
    def train(self):
        pl_trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            precision=PRECISION,
            accelerator="gpu" if self.device_ids else "cpu",
            devices=self.device_ids if self.device_ids else None,
            strategy="single_device" if len(self.device_ids) == 1 else "ddp",
            callbacks=[self.recent_checkpoint_callback, self.best_checkpoint_callback],
            log_every_n_steps=LOG_STEPS
        )
        
        pl_trainer.fit(self.model_wrapper, datamodule=self.dataset_wrapper, ckpt_path=self.resume_ckpt)