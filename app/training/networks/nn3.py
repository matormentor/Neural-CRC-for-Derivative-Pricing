
import pytorch_lightning as pl
import torch

# Combined model NN3 (Training through NN1 -> NN2)
class NN3(pl.LightningModule):
    def __init__(self, nn1: pl.LightningModule, nn2: pl.LightningModule, lr=1e-3):
        super(NN3, self).__init__()
        self.lr = lr
        self.mse_loss = torch.nn.MSELoss()
        self.nn1 = nn1
        self.nn2 = nn2

    def forward(self, nn2_input, nn1_extra_params):
        out1 = self.nn2(nn2_input)
        nn1_input = torch.cat((out1, nn1_extra_params), dim=1)
        return self.nn1(nn1_input)  # NN1 maps parameters to IVS
    
    def training_step(self, batch, batch_idx):
        input1, input2, targets = batch
        predictions = self(input1, input2)
        loss = self.mse_loss(predictions, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input1, input2, targets = batch
        predictions = self(input1, input2)
        loss = self.mse_loss(predictions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input1, input2, targets = batch
        predictions = self(input1, input2)
        loss = self.mse_loss(predictions, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    