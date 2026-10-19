import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import yprov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 8
EPOCHS = 3
DEVICE = "cpu"

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="cellpose-sam-finetuned-hpa", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    metrics_file_type="csv",
    use_compressor=False, 
)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 28 * 28), 
            torch.nn.Linear(28 * 28, 28 * 28), 
            torch.nn.Linear(28 * 28, 10), 
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
        

tform = transforms.Compose([transforms.ToTensor()])

yprov4ml.log_source_code(use_git=True)

yprov4ml.log_context("www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000")
yprov4ml.log_context("Training", "www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000")
yprov4ml.log_context("Validation", "www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000")

mnist_model = MNISTModel().to(DEVICE)
yprov4ml.log_model("parent", mnist_model, context="Training", source="ri-scale/cellpose-sam", is_input=True)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*150))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
yprov4ml.log_dataset("training_data", train_loader, context="Training", source="www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000", other_attrs={"dcterms:identifier": "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000"})

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*50))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
yprov4ml.log_dataset("validation_data", test_loader, context="Validation", source="www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000", other_attrs={"dcterms:identifier": "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000"})

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
yprov4ml.log_param("optimizer", "Adam", is_input=True)
yprov4ml.log_param("epochs", EPOCHS, is_input=True)
yprov4ml.log_param("learning_rate", 1.0e-5, is_input=True)
yprov4ml.log_param("batch_size", BATCH_SIZE, is_input=True)

loss_fn = nn.MSELoss().to(DEVICE)

losses = []
for epoch in range(EPOCHS):
    mnist_model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    
        yprov4ml.log_metric("MSE", loss.item(), context="Training", step=epoch)
        yprov4ml.log_system_metrics("Training", step=epoch)

    mnist_model.eval()
    with torch.no_grad(): 
        # mnist_model.cpu()
        for (x, y) in tqdm(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = mnist_model(x)
            y2 = F.one_hot(y, 10).float()
            loss = loss_fn(y_hat, y2)

            yprov4ml.log_metric("MSE", loss.item(), context="Validation", step=epoch)

yprov4ml.log_model("weights", mnist_model, context="Training", source="ri-scale/cellpose-sam")

yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False, create_card=True)