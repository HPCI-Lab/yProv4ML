import lightning as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append("../yProv4ML")

import yprov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 3
LR = 0.0002

# The logger starts the run (start_run) in its constructor.
# end_on_finalize=False: Lightning finalizes loggers after fit() AND test(),
# so we end the run ourselves after testing.
logger = yprov4ml.yProvMLLogger(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name",
    provenance_save_dir="prov",
    save_after_n_logs=100,
    create_ro_crate=True,
)

callback = yprov4ml.yProvMLCallback(
    # model_attrs={"dcterms:identifier": "ri-scale/cellpose-sam"},
    # dataset_attrs={"dcterms:identifier": "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000"},
)


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(torch.nn.Linear(28 * 28, 10))

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

    # Plain self.log: the logger adds the epoch and the Training/Validation/Test context.
    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("MSE_T", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("MSE_V", loss)
        return loss

    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("MSE_test", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)


# One-time params, linked as inputs of the first training activity
yprov4ml.log_param("optimizer", "Adam", is_input_to="Training_0")
yprov4ml.log_param("epochs", EPOCHS, is_input_to="Training_0")
yprov4ml.log_param("learning_rate", LR, is_input_to="Training_0")
yprov4ml.log_param("batch_size", BATCH_SIZE, is_input_to="Training_0")

tform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])
yprov4ml.log_param("dataset_transformation", str(tform), is_input_to="Training_0")

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
val_ds = Subset(train_ds, range(BATCH_SIZE * 1))
train_ds = Subset(train_ds, range(BATCH_SIZE * 10))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

mnist_model = MNISTModel()

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=EPOCHS,
    logger=logger,
    log_every_n_steps=1,       # small dataset: log every step
    enable_checkpointing=False,
    callbacks=[callback], 
)
trainer.fit(mnist_model, train_loader, val_dataloaders=val_loader)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE * 2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
yprov4ml.log_dataset("test_dataset", test_loader, is_input_to="Test_0", is_output_to="Download")

trainer.test(mnist_model, test_loader)

# Write the provenance graphs
logger.finalize(status="Success")