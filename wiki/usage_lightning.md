
# Example of usage with PyTorch Lightning

This section provides an example of how to use Prov4ML with PyTorch Lightning.

### ProvMLLogger Class

The ProvMLLogger class is a custom logger that facilitates the logging of machine learning experiment data, such as metrics and hyperparameters, with additional features for versioning and directory management.

#### Example

```python
trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    logger=[prov4ml.ProvMLLogger(name="mnist_model")],
    enable_checkpointing=False, 
)

# in the LightningModule
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    self.log("train_loss", loss) # if the ProvMLLogger is used
    return loss
```

Any call inside the trainer `self.log` will be logged to the provenance graph.

### Logging inside the LightningModule

In any lightning module the calls to `train_step`, `validation_step`, and `test_step` can be overridden to log the necessary information.

```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    prov4ml.log_metric("MSE_train", loss, prov4ml.Context.TRAINING, step=self.current_epoch)
    prov4ml.log_flops_per_batch("train_flops", self, batch, prov4ml.Context.TRAINING,step=self.current_epoch)
    return loss
```

This will log the mean squared error and the number of flops per batch for each the training step.

Alternatively, the `on_train_epoch_end` method can be overridden to log information at the end of each epoch.

```python
def on_train_epoch_end(self) -> None:
    prov4ml.log_metric("epoch", self.current_epoch, prov4ml.Context.TRAINING, step=self.current_epoch)
    prov4ml.save_model_version(self, f"model_version_{self.current_epoch}", prov4ml.Context.TRAINING, step=self.current_epoch)
    prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
    prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
    prov4ml.log_current_execution_time("train_epoch_time", prov4ml.Context.TRAINING, self.current_epoch)
```