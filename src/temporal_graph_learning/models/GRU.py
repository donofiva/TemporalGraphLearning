import torch
import torch.nn as nn
import lightning as lt
import torch.optim as opt
import torch.nn.functional as fn


class GRU(lt.LightningModule):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        layers: int,
        learning_rate: float
    ):

        # Initialize superclass
        super().__init__()

        # Data shapes
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size

        # Model configuration
        self._layers = layers

        # Optimizer configuration
        self._learning_rate = learning_rate

        # Define model
        self._gru = nn.GRU(self._input_size, self._hidden_size, batch_first=True, num_layers=self._layers)
        self._fc = nn.Linear(self._hidden_size, self._output_size)

    def forward(self, x):

        # GRU pass
        x, _ = self._gru(x)

        # FC pass
        x = self._fc(x[:, -1, :])

        # Reshape output
        return x.view(
            x.size(0),
            -1,
            self._output_size
        )

    def configure_optimizers(self):
        return opt.Adam(self.parameters(), lr=self._learning_rate)

    def training_step(self, batch, index):

        # Expand batch
        inputs, masks_input, masks_output, outputs = batch

        # Move batch to device
        inputs = inputs.to(self.device)
        masks_input = masks_input.to(self.device)
        masks_output = masks_output.to(self.device)
        outputs = outputs.to(self.device)

        # Mask inputs
        inputs_masked = torch.where(masks_input > 0.5, inputs, -1)
        inputs_masked[:, :, -2:] = inputs[:, :, -2:]

        # Predict outputs
        outputs_predicted = self(inputs_masked)

        # Mask outputs
        outputs_predicted_masked = outputs_predicted[masks_output > 0.5]
        outputs_masked = outputs[masks_output > 0.5]

        # Compute loss
        loss = fn.l1_loss(outputs_masked, outputs_predicted_masked)

        # Log
        self.log(
            "l1_loss_training",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss

    def validation_step(self, batch, index):

        # Expand batch
        inputs, masks_input, masks_output, outputs = batch

        # Move batch to device
        inputs = inputs.to(self.device)
        masks_input = masks_input.to(self.device)
        masks_output = masks_output.to(self.device)
        outputs = outputs.to(self.device)

        # Mask inputs
        inputs_masked = torch.where(masks_input > 0.5, inputs, -1)
        inputs_masked[:, :, -2:] = inputs[:, :, -2:]

        # Predict outputs
        outputs_predicted = self(inputs_masked)

        # Mask outputs
        outputs_predicted_masked = outputs_predicted[masks_output > 0.5]
        outputs_masked = outputs[masks_output > 0.5]

        # Compute loss
        loss = fn.l1_loss(outputs_masked, outputs_predicted_masked)

        # Log
        self.log(
            "l1_loss_validation",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

        return loss
