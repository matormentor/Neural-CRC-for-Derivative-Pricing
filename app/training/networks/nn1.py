import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torchmetrics import R2Score

# NN1 with residual connections (mapping parameters -> IVS)
class NN1Residual(pl.LightningModule):
    def __init__(self, input_dim=41, hidden_dim=1024, output_dim=130, lr=1e-3):
        super(NN1Residual, self).__init__()
        self.save_hyperparameters()
        self.mse_loss = nn.MSELoss()
        self.r2Score = R2Score(adjusted=input_dim)  # Assuming all regressors independent
        self.activation = nn.ELU()
        self.lr = lr
        
        # Input Layer
        self.input_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # We have 41 financial parameters,
            self.activation)
        
        # Hidden block 
        self.hidden_residual_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)  # 130 IVS points

    def forward(self, x):
        # Input
        residual = self.input_head(x)
        
        # Residual Block
        x = self.hidden_residual_block(residual) + residual
        
        # Output with final activation?
        return self.output_layer(self.activation(x))
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.mse_loss(predictions, targets)
        r2s = self.r2Score(predictions, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("adj_train_r2_score", r2s, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.mse_loss(predictions, targets)
        r2s = self.r2Score(predictions, targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("adj_val_r2_score", r2s, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.mse_loss(predictions, targets)
        r2s = self.r2Score(predictions, targets)
        self.log("test_loss", loss, prog_bar=True)
        self.log("adj_test_r2_score", r2s, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)

