from torch.utils.data import Dataset, DataLoader, RandomSampler
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import multiprocessing
import pytorch_lightning as pl
from torchvision.datasets import MNIST
import numpy as np
import torch

PathLike = Path | str

def random_color(im):
    hue = 360*np.random.rand()
    d = (im *(hue%60)/60)
    im_min, im_inc, im_dec = torch.zeros_like(im), d, im - d
    c_im = torch.zeros((3, im.shape[1], im.shape[2]))
    H = round(hue/60) % 6    
    cmap = [[0, 3, 2], [2, 0, 3], [1, 0, 3], [1, 2, 0], [3, 1, 0], [0, 1, 2]]
    return torch.cat((im, im_min, im_dec, im_inc), dim=0)[cmap[H]]

class CelebDataset(Dataset):
    def __init__(
        self,
        img_dir: PathLike,
        train: bool = True,
        val: bool = False,
        test: bool = False,
        transform=None,
    ) -> None:
        super().__init__()
        img_dir = Path(img_dir)

        # TODO: refactor code
        if train:
            path_to_male = img_dir / "Train" / "Male"
            path_to_female = img_dir / "Train" / "Female"
        elif val:
            path_to_male = img_dir / "Validation" / "Male"
            path_to_female = img_dir / "Validation" / "Female"

        elif test:
            path_to_male = img_dir / "Test" / "Male"
            path_to_female = img_dir / "Test" / "Female"

        else:
            assert False, "choose train, val or test mode"

        imgs_male = list(path_to_male.glob("*"))
        imgs_female = list(path_to_female.glob("*"))
        self.min_len = min(len(imgs_male), len(imgs_female))
        self.imgs_male = imgs_male[: self.min_len]
        self.imgs_female = imgs_female[: self.min_len]
        self.transform = transform

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        image_male = Image.open(self.imgs_male[idx])
        image_female = Image.open(self.imgs_female[idx])

        if self.transform:
            image_male = self.transform(image_male)
            image_female = self.transform(image_female)

        return image_male, image_female


class CelebDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir: PathLike,
        batch_size: int = 64,
        device: str = "cuda",
        num_workers: int = multiprocessing.cpu_count(),
    ):
        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.sampler = RandomSampler()

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.celeb_train = CelebDataset(
                self.img_dir, train=True, transform=self.transform
            )
            self.celeb_val = CelebDataset(
                self.img_dir, val=True, transform=self.transform
            )
        # Assign test dataset
        if stage == "test" or stage is None:
            self.celeb_test = CelebDataset(
                self.img_dir, test=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.celeb_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
            sampler=RandomSampler(
                self.celeb_train, replacement=True, num_samples=self.batch_size * 1000
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeb_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
            sampler=RandomSampler(
                self.celeb_val, replacement=True, num_samples=self.batch_size * 1000
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeb_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
            sampler=RandomSampler(
                self.celeb_test, replacement=True, num_samples=self.batch_size * 1000
            ),
        )


class MnistDataset(Dataset):
    def __init__(
        self,
        img_dir: PathLike,
        train: bool = True,
        val: bool = False,
        test: bool = False,
        transform=None,
    ) -> None:
        super().__init__()
        img_dir = Path(img_dir)

        # TODO: refactor code
        if train:
            path_to_1 = img_dir / "Train" / "1"
            path_to_2 = img_dir / "Train" / "2"
        elif val:
            path_to_1 = img_dir / "Validation" / "1"
            path_to_2 = img_dir / "Validation" / "2"

        elif test:
            path_to_1 = img_dir / "Test" / "1"
            path_to_2 = img_dir / "Test" / "2"

        else:
            assert False, "choose train, val or test mode"

        imgs_1 = list(path_to_1.glob("*"))
        imgs_2 = list(path_to_2.glob("*"))
        self.min_len = min(len(imgs_1), len(imgs_2))
        self.imgs_1 = imgs_1[: self.min_len]
        self.imgs_2 = imgs_2[: self.min_len]
        self.transform = transform

    def __len__(self):
        return self.min_len

    def __getitem__(self, idx):
        image_1 = Image.open(self.imgs_1[idx])
        image_2 = Image.open(self.imgs_2[idx])
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2


class MnistDataModule(pl.LightningDataModule):
    def __init__(
        self,
        img_dir: PathLike,
        batch_size: int = 64,
        device: str = "cuda",
        num_workers: int = 1,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        
        self.transform = T.Compose([
            T.Resize(16),
            T.ToTensor(),
            random_color,
            T.Normalize([0.5],[0.5])
        ])

    def prepare_data(self):
        # download
        MNIST(self.img_dir, train=True, download=True)
        MNIST(self.img_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.mnist_train = MnistDataset(
                self.img_dir, train=True, transform=self.transform
            )
            self.mnist_val = MnistDataset(
                self.img_dir, val=True, transform=self.transform
            )
        # Assign test dataset
        if stage == "test" or stage is None:
            self.mnist_test = MnistDataset(
                self.img_dir, test=True, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
            sampler=RandomSampler(
                self.mnist_train, replacement=True, num_samples=self.batch_size * 1000
            ),
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
            sampler=RandomSampler(
                self.mnist_val, replacement=True, num_samples=self.batch_size * 1000
            ),
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
            sampler=RandomSampler(
                self.mnist_test, replacement=True, num_samples=self.batch_size * 1000
            ),
        )
