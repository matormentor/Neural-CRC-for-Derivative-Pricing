import pytorch_lightning as L
import torch
from torchmetrics import R2Score

import app.training.networks as net

# Combined model NN3 (Training through NN1 -> NN2)
class NN3(L.LightningModule):
    def __init__(self, nn1: net.NN1Residual, nn2: net.NN2, lr=1e-3):
        super(NN3, self).__init__()
        self.save_hyperparameters(ignore=['nn1', 'nn2'])
        self.lr = lr
        self.mse_loss = torch.nn.MSELoss()
        self.r2Score = R2Score(adjusted=161)  # Total independent regressors
        self.nn1 = nn1
        self.nn2 = nn2
        self.nn1.eval()  # Ensures BatchNorm behave in evaluation mode safety, maybe not necessary
        for param in self.nn1.parameters():
            param.requires_grad = False 

    def forward(self, nn2_input, nn1_extra_params):
        out_nn2 = self.nn2(nn2_input)
        nn1_input = torch.cat((nn1_extra_params, out_nn2), dim=1)
        return self.nn1(nn1_input)  # NN1 maps parameters to IVS
    
    def training_step(self, batch, batch_idx):
        input1, input2, targets = batch
        predictions = self(input1, input2)
        loss = self.mse_loss(predictions, targets)
        r2s = self.r2Score(predictions, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("adj_train_r2_score", r2s, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input1, input2, targets = batch
        predictions = self(input1, input2)
        loss = self.mse_loss(predictions, targets)
        r2s = self.r2Score(predictions, targets)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("adj_val_r2_score", r2s, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        input1, input2, targets = batch
        predictions = self(input1, input2)
        loss = self.mse_loss(predictions, targets)
        r2s = self.r2Score(predictions, targets)
        self.log("nn3_test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("adj_test_r2_score", r2s, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr) # try weight decay
    