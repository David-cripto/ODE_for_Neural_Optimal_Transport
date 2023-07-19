from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
from torchdiffeq import odeint
from torchvision.models import resnet50, ResNet50_Weights
import typing as tp
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision

class ODEModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO: maybe pass depth to kwargs
        # TODO: UNet 
        depth = 2
        self.nn = nn.Sequential(
            *[
                nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.BatchNorm2d(3), nn.ReLU())
                for _ in range(depth)
            ]
        )

    def forward(self, t, x):
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
        return odeint(y0=x, t=self.t.to(x), func=self.ode_func)[1]
    
class SimpleFunc(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*[i for i in list(resnet.children())[:-1]])
        dim = 2048
        head = nn.ModuleList([])
        while dim != 64:
            head.extend([nn.Sequential(nn.Linear(dim, dim//2), nn.ReLU())])
            dim //= 2 
        head.extend([nn.Sequential(nn.Linear(dim, 1), nn.ReLU())])
        self.head = nn.Sequential(*head)
    def forward(self, x):
        if x.shape[0] == 1:
            return self.head(self.model(x).squeeze().unsqueeze(0))
            
        return self.head(self.model(x).squeeze())

class NeuralTransfer(pl.LightningModule):
    def __init__(self, TransferFunc, RegulaizerFunc, lr: float = 0.0002) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        
        self.transfer = TransferFunc()
        self.regularize = RegulaizerFunc()
        
    def forward(self, x):
        return self.transfer(x)
    
    # TODO: 1:10 --- discriminator vs generator
    # TODO: profile one iteration
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs_p, imgs_q = batch
        optimizer_g, optimizer_d = self.optimizers()
        
        self.toggle_optimizer(optimizer_g)
        generated_imgs = self(imgs_p)
        
        true_imgs = imgs_p[:2]
        grid = torchvision.utils.make_grid(true_imgs)
        self.logger.experiment.add_image("true images", grid, self.current_epoch)
        
        sample_imgs = generated_imgs[:2]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)
        
        g_loss = F.mse_loss(generated_imgs, imgs_p) - self.regularize(generated_imgs).mean()
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        d_loss = self.regularize(self(imgs_p).detach()).mean() - self.regularize(imgs_q).mean()
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)
        
    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_g = torch.optim.Adam(self.transfer.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.regularize.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    