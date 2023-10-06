import os
import gc
import numpy as np
import random
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from mnist_dataloaders import prepare_mnist_data

from mpl_pl_model import MPLNetWrapper
from config import CONFIG

class PeriodicCheckpoint(ModelCheckpoint):
    '''
    Save checkpoints every self.every epoch
    '''
    def __init__(self, every: int):
        super().__init__()
        self.every = every

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.current_epoch % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.current_epoch}.ckpt"
            prev = (
                Path(self.dirpath) / f"latest-{pl_module.current_epoch - self.every}.ckpt"
            )
            trainer.save_checkpoint(current)
            prev.unlink(missing_ok=True)


def fit_and_validate(train_loader, validation_loader, accelerator) -> float:

    model = MPLNetWrapper(in_dim=CONFIG['in_dim'], out_dim=CONFIG['out_dim'], lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'], batch_size=CONFIG['batch_size'])
    
    csv_logger = CSVLogger(save_dir=os.path.join(CONFIG['logs_dir'], "csv_logs"))    

    trainer_kwargs = dict()
    trainer_kwargs["accelerator"] = accelerator

    if CONFIG["restore_from_checkpoint"] and os.path.isfile(CONFIG["checkpoint_path"]):
        trainer_kwargs['resume_from_checkpoint'] = CONFIG["checkpoint_path"]

    # Best by "val_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=None,
        monitor="val_loss",
        mode="min",
        save_top_k=1
        )
    period_checkpoint_callback = PeriodicCheckpoint(every=CONFIG['periodic_checkpoint_rate'])

    trainer = pl.Trainer(default_root_dir=os.path.join(CONFIG['logs_dir'], "train_logs"),
                         max_epochs=CONFIG["num_epochs"], 
                         log_every_n_steps=5,
                         logger=[csv_logger],
                         enable_checkpointing=True,
                         callbacks=[checkpoint_callback,
                                    period_checkpoint_callback],
                         **trainer_kwargs)

    hyperparameters = CONFIG
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, train_loader, validation_loader)
    trainer.validate(model, validation_loader)

    csv_logger.save()


def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    accelerator = 'cpu'
    if torch.cuda.is_available():
        # Cuda maintenance
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(CONFIG['seed'])
        accelerator = 'gpu'

    print(f"Accelerator: {accelerator}")

    if not os.path.isdir(CONFIG['data_folder']):
        os.makedirs(CONFIG['data_folder'])

    train_loader, validation_loader = prepare_mnist_data(data_folder=CONFIG['data_folder'], batch_size=CONFIG['batch_size'])
    fit_and_validate(train_loader, validation_loader, accelerator)


if __name__ == "__main__":
    main()