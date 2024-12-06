import pytorch_lightning as L
from torch.utils.data import random_split, Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler
from joblib import cpu_count

import polars as pl

class IVS_Dataset(Dataset):
    def __init__(self, features, targets):
        """
        Dataset to handle IVS data.
        Args:
            features (Tensor): Normalized feature array.
            targets (Tensor): Target array.
        """
        self.features = features
        self.targets = targets
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    

class IVS_DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str = "./data/bates_iv_surface_dataset.parquet", batch_size=1000, test_split=0.2, val_split=0.1, random_seed=42, ivs_size=130):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.random_seed = random_seed
        self.ivs_size = ivs_size

    def prepare_data(self):
        # Load and preprocess the dataset
        self.data = pl.read_parquet(self.data_dir)
        # If filtering Required
        # self.data = self.data.filter(
        #                     ~self.data["implied_vol_surface"].list.eval(pl.element().is_nan()).list.any()
        #                 )

        # Extract features and targets
        self.features = self.data.drop("implied_vol_surface").to_numpy()
        self.targets = torch.from_numpy(self.data["implied_vol_surface"].cast(pl.Array(pl.Float32, self.ivs_size)).to_numpy())  # Known Size of 130        
        
        # Normalize features to (0, 1)  
        # normalize = self._get_normalize_transform(features=features)
        self.features = torch.from_numpy(MinMaxScaler().fit_transform(self.features))
        
        self.dataset = IVS_Dataset(features=self.features, targets=self.targets)
        
    def setup(self, stage=None):
        full_dataset = self.dataset
        
        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset,
            [1 - self.val_split - self.test_split, self.val_split, self.test_split],
            generator=torch.Generator().manual_seed(self.random_seed)
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size)
    