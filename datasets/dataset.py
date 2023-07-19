from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
import multiprocessing
import pytorch_lightning as pl


PathLike = Path | str


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
        self.imgs_male = imgs_male[:self.min_len]
        self.imgs_female = imgs_female[:self.min_len]
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
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeb_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeb_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            pin_memory_device=self.device,
        )