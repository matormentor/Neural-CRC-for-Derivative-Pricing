import pytorch_lightning as L
from torch.utils.data import random_split, Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler
import polars as pl


class DualInput_IVS_Dataset(Dataset):
    def __init__(self, features1, features2, targets):
        """
        Dataset to handle IVS data with two inputs and a target.
        Args:
            features1 (Tensor): Normalized feature array for input 1.
            features2 (Tensor): Normalized feature array for input 2.
            targets (Tensor): Target array.
        """
        self.features1 = features1
        self.features2 = features2
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features1[idx], self.features2[idx], self.targets[idx]


class DualInput_IVS_DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data/ivs_dataset.parquet", batch_size=1000, test_split=0.2, val_split=0.1, random_seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.random_seed = random_seed
        # Normalize Features separately
        self.scaler1 = MinMaxScaler()  # To be accessed for further Testing in the recalibration algorithm 
        self.scaler2 = MinMaxScaler()  # To be accessed for further Testing in the recalibration algorithm

    def prepare_data(self):
        self.data = pl.read_parquet(self.data_dir)

        input1_columns = ["theta", "sigma", "rho"]
        input2_drops = [f"nu_{i}" for i in range(1, 6)] + [f"delta_{i}" for i in range(1, 6)] + ["implied_vol_surface"]

        # Input2
        features2 = self.data.drop(input2_drops).to_numpy()
        
        # scale only the other parameters, leave IVS unchanged
        part_features1 = torch.from_numpy(
            self.scaler1.fit_transform(self.data.select(input1_columns).to_numpy()))
        
        # Input1 includes the first three columns + implied_vol_surface
        self.features1 = torch.cat(
            [part_features1,
             torch.from_numpy(self.data['implied_vol_surface'].to_numpy())],
            dim=1
        )
        self.features2 = torch.from_numpy(
            self.scaler2.fit_transform(features2))
        
        self.targets = torch.from_numpy(self.data["implied_vol_surface"].to_numpy())

        # Save processed dataset
        self.dataset = DualInput_IVS_Dataset(
            features1=self.features1, features2=self.features2, targets=self.targets
        )

    def setup(self, stage=None):

        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset,
            [1 - self.val_split - self.test_split, self.val_split, self.test_split],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
