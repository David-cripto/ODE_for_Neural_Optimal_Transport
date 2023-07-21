from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from torchvision.models import resnet50, ResNet50_Weights
import typing as tp
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision
from dpipe import layers


class ODEModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        structure = [
            [[80, 80, 80], [160, 160, 160]],
            [[80, 80, 80], [160, 80, 80]],
            [[80, 128, 128], [256, 128, 80]],
            [[128, 128, 256], [512, 256, 128]],
            [256, 512, 256],
        ]
        self.nn = nn.Sequential(
            nn.Conv2d(3, 80, 3, padding=1),
            layers.FPN(
                layers.ResBlock2d,
                nn.MaxPool2d(2),
                nn.Identity(),
                structure=structure,
                merge=lambda left, down: torch.cat(
                    layers.interpolate_to_left(left, down, 1), 1
                ),
                kernel_size=3,
                padding=1,
            ),
            nn.Conv2d(160, 3, 3, padding=1),
        )
        self.num_solver_steps = 0

    def forward(self, t, x):
        self.num_solver_steps += 1
        # TODO: positional + condtional instance normalization(diffusions)
        return self.nn(x)


class ODEBlock(nn.Module):
    def __init__(
        self,
        time: tp.Optional[float] = 1,
    ) -> None:
        super().__init__()
        self.t = torch.tensor([0, time])
        self.ode_func = ODEModel()

    def forward(self, x):
        # TODO: restrict number steps of solvers
        # TODO: track number steps of solvers
        return odeint_adjoint(y0=x, t=self.t.to(x), func=self.ode_func, method="euler")[
            1
        ]


class SimpleFunc(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*[i for i in list(resnet.children())[:-1]])
        dim = 2048
        head = nn.ModuleList([])
        while dim != 64:
            head.extend([nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU())])
            dim //= 2
        head.extend([nn.Sequential(nn.Linear(dim, 1), nn.ReLU())])
        self.head = nn.Sequential(*head)

    def forward(self, x):
        if x.shape[0] == 1:
            return self.head(self.model(x).squeeze().unsqueeze(0))

        return self.head(self.model(x).squeeze())


class NeuralTransfer(pl.LightningModule):
    def __init__(self, TransferFunc, RegulaizerFunc, lr: float = 0.00002) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.transfer = TransferFunc()
        self.regularize = RegulaizerFunc()

        # TODO: Maybe somewhere has better idea how to realize that
        self.n_train = 0
        self.n_limit = 11

    def forward(self, x):
        return self.transfer(x)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs_p, imgs_q = batch
        optimizer_g, optimizer_d = self.optimizers()

        if self.n_train < self.n_limit:
            self.toggle_optimizer(optimizer_g)
            generated_imgs = self(imgs_p)
            g_loss = (
                F.mse_loss(generated_imgs, imgs_p)
                - self.regularize(generated_imgs).mean()
            )
            self.log("g_loss", g_loss, prog_bar=True)
            self.manual_backward(g_loss)
            optimizer_g.step()
            optimizer_g.zero_grad()
            self.untoggle_optimizer(optimizer_g)
            self.n_train += 1
        else:
            self.toggle_optimizer(optimizer_d)
            d_loss = (
                self.regularize(self(imgs_p).detach()).mean()
                - self.regularize(imgs_q).mean()
            )
            self.log("d_loss", d_loss, prog_bar=True)
            self.manual_backward(d_loss)
            optimizer_d.step()
            optimizer_d.zero_grad()
            self.untoggle_optimizer(optimizer_d)
            self.n_train = 0

    def on_train_batch_end(
        self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.log(
            "num_solver_step", self.transfer.ode_func.num_solver_steps, prog_bar=True
        )
        self.transfer.ode_func.num_solver_steps = 0

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        if batch_idx == 0:
            imgs_p, imgs_q = batch
            true_imgs = imgs_p[:2]
            grid = torchvision.utils.make_grid(true_imgs)
            self.logger.experiment.add_image("true images", grid, self.current_epoch)

            generated_imgs = self(imgs_p)
            sample_imgs = generated_imgs[:2]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image(
                "generated_images", grid, self.current_epoch
            )

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.transfer.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.regularize.parameters(), lr=lr)
        return [opt_g, opt_d], []
