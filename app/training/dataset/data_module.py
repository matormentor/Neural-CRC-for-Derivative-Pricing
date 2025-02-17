import pytorch_lightning as L
from torch.utils.data import random_split, Dataset, DataLoader
import torch
from sklearn.preprocessing import MinMaxScaler
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
    def __init__(self, data_dir: str = "./data/ivs_dataset.parquet", batch_size=1000, test_split=0.2, val_split=0.1, random_seed=42, ivs_size=130):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.random_seed = random_seed
        self.ivs_size = ivs_size
        self.min_max_scaler = MinMaxScaler()  # To be accessed for further Testing in the recalibration algorithm

    def prepare_data(self):
        # Load and preprocess the dataset
        self.data = pl.read_parquet(self.data_dir)

        # Extract and Normalize features to (0, 1)  
        self.features = torch.from_numpy(self.min_max_scaler.fit_transform(self.data.drop("implied_vol_surface").to_numpy()))
        
        self.targets = torch.from_numpy(self.data["implied_vol_surface"].to_numpy())  # Known Size of 130        
        
        self.dataset = IVS_Dataset(features=self.features, targets=self.targets)
        
    def setup(self, stage=None):
        self.train_set, self.val_set, self.test_set = random_split(
            self.dataset,
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
    