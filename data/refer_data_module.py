from argparse import Namespace
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from utils import transforms as T
from data.dataset_refer_bert import ReferDataset


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def get_dataset(split: str, args, eval_mode: bool):
    transform = get_transform(args)
    ds = ReferDataset(args,
                      split=split,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=eval_mode)
    num_classes = 2
    return ds, num_classes


class ReferDataModule(LightningDataModule):
    def __init__(self, args: Namespace):
        LightningDataModule.__init__(self)
        self.args = args

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = -1

    def setup(self, stage: Optional[str] = None):
        # Assign train/val datasets for use in data-loaders.
        if stage == 'fit' or stage is None:
            self.train_dataset, self.num_classes = get_dataset(split='train', args=self.args, eval_mode=False)
            self.val_dataset, _ = get_dataset(split='val', args=self.args, eval_mode=False)

        # Assign test dataset for use in data-loader(s).
        if stage == 'test' or stage is None:
            self.test_dataset, _ = get_dataset(split='test', args=self.args, eval_mode=True)

    @property
    def batch_size(self) -> int:
        return self.args.batch_size

    @property
    def workers(self) -> int:
        return self.args.workers

    @property
    def pin_mem(self) -> bool:
        return self.args.pin_mem

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=self.pin_mem,
            drop_last=True)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.pin_mem,
            drop_last=False)

    def __len__(self):
        return len(self.train_dataloader())
