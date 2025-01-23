import torch.nn as nn
import torch
import pytorch_lightning as pl

class StretchedSigmoid(nn.Module):
    """
    Custom activation function to stretch the sigmoid function to the given range.
    """
    def __init__(self, min_value, max_value):
        super(StretchedSigmoid, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return self.min_value + (self.max_value - self.min_value) * torch.sigmoid(x)
    

class NN2(pl.LightningModule):
    def __init__(self,input_size=133, hidden_size=1024, output_size=10, lr=0.001):
        """
        A feedforward neural network with a residual block and additional processing layers. 
        Designed to output the means and variances for the Levy processes
        
        This network includes:
        - An input layer with batch normalization and ELU activation.
        - A residual block with four hidden layers, each followed by batch normalization and ELU activation.
        - Additional processing layers after the residual block, including batch normalization and ELU activation.
        - Two separate linear layers for feature reduction into means and variances before concatenating into the final output.
        
        **IMPORTANT:** The output size must be even.

        Example:
        --------
        >>> nn2 = NN2()
        >>> dummy_input = torch.randn(1, 133)
        >>> nn2.eval() # For doing only one sample
        >>> output = nn2(dummy_input)
        >>> print("Output shape:", output.shape)  # Should be (1, 10)
        """
        super(NN2, self).__init__()
        assert output_size % 2 == 0, f"output size {output_size}, is not even!"
        self.save_hyperparameters()
        self.hidden_size = hidden_size
        self.mse_loss = nn.MSELoss()
        self.lr = lr
        
        # Input layer
        self.input_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU()
        )
        
        # Four hidden layers
        self.hidden_residual_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ELU(),
        )
        
        # Second interpretation
        # self.hidden_residual_block = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.ELU()
        # )
        
        # Additional layers after the main residual connection
        self.output_layer1 = nn.Linear(hidden_size, output_size)
        self.batch_norm2 = nn.BatchNorm1d(output_size)
        
        # Final concatenation layers
        self.nu_layer = nn.Sequential(
            nn.Linear(output_size, output_size//2),
            StretchedSigmoid(-0.3, 0.3)
        )
        
        self.delta_layer = nn.Sequential(
            nn.Linear(output_size, output_size//2),
            StretchedSigmoid(0.2, 0.3)
        )

    def forward(self, x):
        # Input processing
        residual = self.input_head(x)

        # Residual connection
        x = self.hidden_residual_block(residual) + residual 
        
        # Additional processing layers
        x = self.output_layer1(nn.ELU()(x))
        x = self.batch_norm2(x)
        
        # Dense layers before concatenation
        nus = self.nu_layer(x)
        delta = self.delta_layer(x)
        
        output = torch.cat((nus, delta), dim=1)
        
        return output

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.mse_loss(predictions, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.mse_loss(predictions, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.mse_loss(predictions, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)