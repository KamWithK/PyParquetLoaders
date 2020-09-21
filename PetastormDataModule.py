from pathlib import Path
from pytorch_lightning import LightningModule, LightningDataModule
from petastorm import make_batch_reader
from petastorm.transform import TransformSpec
from petastorm.pytorch import DataLoader

class PetastormDataModule(LightningDataModule):
    def __init__(self, train_path, val_path, batch_size=16, num_minibatchs=8, num_workers=4, transform_spec=None):
        super().__init__()
        self.train_path, self.val_path = Path(train_path).absolute().as_uri(), Path(val_path).absolute().as_uri()
        self.batch_size, self.num_minibatchs, self.num_workers = batch_size, num_minibatchs, num_workers

        self.transform_spec = transform_spec

    # Called once per GPU
    def setup(self, stage=None):
        self.train_dataset = make_batch_reader(self.train_path, workers_count=self.num_workers, transform_spec=self.transform_spec)
        self.val_dataset = make_batch_reader(self.val_path, workers_count=self.num_workers, transform_spec=self.transform_spec)

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)
