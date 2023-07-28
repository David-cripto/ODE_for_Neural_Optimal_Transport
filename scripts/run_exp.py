import pytorch_lightning as pl
import os

os.chdir("/homes/kek/ODE_for_Neural_Optimal_Transport")

from models.models import NeuralTransfer, ODEBlock, SimpleFunc
from datasets.dataset import CelebDataModule

model = NeuralTransfer(ODEBlock, SimpleFunc)
dm = CelebDataModule("/homes/kek/ODE_for_Neural_Optimal_Transport/datasets/Dataset")
tb_logger = pl.loggers.TensorBoardLogger(save_dir="logs/")

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=15,
    logger=tb_logger
)
trainer.fit(model, dm)


