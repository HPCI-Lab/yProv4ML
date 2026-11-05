import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append("yprov4ml")
import yprov4ml
import yprov4dv

PATH_DATASETS = "./data"
BATCH_SIZE = 8
EPOCHS = 3
DEVICE = "cpu"

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="riscale-usecase", 
    provenance_save_dir="prov",
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

mnist_model = MNISTModel().to(DEVICE)
yprov4ml.log_model("parent", mnist_model, is_input_to="Training_0", is_output_to="Download")
yprov4ml.log_param("learning_rate", 1.0e-5, is_input_to="Training_0")

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*150))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*50))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
yprov4ml.log_param("batch_size", BATCH_SIZE, is_input_to="Training_0")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
yprov4ml.log_param("optimizer", "Adam", is_input_to="Training_0")
yprov4ml.log_param("epochs", EPOCHS, is_input_to="Training_0")

loss_fn = nn.MSELoss().to(DEVICE)

losses = []
for epoch in range(EPOCHS):
    yprov4ml.log_dataset("validation_data", test_loader, is_input_to=f"Validation_{epoch}", is_output_to="Download")
    yprov4ml.log_dataset("training_data", train_loader, is_input_to=f"Training_{epoch}", is_output_to="Download")
    
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
    
        yprov4ml.log_metric(f"MSE_T_{epoch}", loss.item(), is_output_to=f"Training_{epoch}", step=epoch, is_input_to="results_visualization")

    mnist_model.eval()
    yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/checkpoint_{epoch}", mnist_model, is_input_to=f"Validation_{epoch}")
    with torch.no_grad(): 
        for (x, y) in tqdm(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_hat = mnist_model(x)
            y2 = F.one_hot(y, 10).float()
            loss = loss_fn(y_hat, y2)

            yprov4ml.log_metric(f"MSE_V_{epoch}", loss.item(), is_output_to=f"Validation_{epoch}", step=epoch, is_input_to="results_visualization")

    if epoch+1 < EPOCHS: 
        yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/checkpoint_{epoch}", mnist_model, is_input_to=f"Training_{epoch+1}", is_output_to=f"Training_{epoch}")
    else: 
        yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/checkpoint_{epoch}", mnist_model, is_output_to=f"Training_{epoch}")

weights_path = f"{yprov4ml.get_artifacts_dir()}/weights"
yprov4ml.log_model(weights_path, mnist_model, is_output_to=f"Training_{EPOCHS-1}")


inference_dataset_id = "https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD0000"
yprov4ml.log_dataset("inference_data", test_loader, is_input_to="Inference_0", is_output_to="Download")
yprov4ml.log_model(weights_path, mnist_model, is_input_to="Inference_0")

mnist_model.eval()
inference_losses = []
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Inference Phase"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = mnist_model(x)
        y_onehot = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y_onehot)
        inference_losses.append(loss.item())
        yprov4ml.log_metric("MSE_Inference", loss.item(), is_output_to="Inference_0", is_input_to="results_visualization")



yprov4dv.start_run(
    run_name="results_visualization", 
    provenance_directory=f"{yprov4ml.get_artifacts_dir()}/results_visualization",
)

# 1. Save Training Loss Plot
plot_loss_path = "training_loss_plot.png"
plt.figure(figsize=(8, 5))
plt.plot(losses, label="Training Loss", alpha=0.6)
plt.title("Training Loss Progression")
plt.xlabel("Batch Steps")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.savefig(plot_loss_path, bbox_inches="tight")
plt.close()

# 2. Save Test Accuracy / Validation Loss Plot (Additional Viz)
plot_acc_path = "test_accuracy_plot.png"
plt.figure(figsize=(8, 5))
plt.plot(inference_losses, label="Inference / Validation Loss", color="orange", alpha=0.6)
plt.title("Test / Inference Evaluation Progress")
plt.xlabel("Batch Steps")
plt.ylabel("Loss / Error")
plt.legend()
plt.grid(True)
plt.savefig(plot_acc_path, bbox_inches="tight")
plt.close()

yprov4dv.end_run()

yprov4ml.log_artifact("plot_reproducibility_package", 
                      f"{yprov4ml.get_artifacts_dir()}/results_visualization_0", 
                      is_output_to="results_visualization")
yprov4ml.log_artifact("training_loss_plot", 
                      f"{yprov4ml.get_artifacts_dir()}/results_visualization_0/outputs/{plot_loss_path}", 
                      is_output_to="results_visualization")
yprov4ml.log_artifact("test_accuracy_plot", 
                      f"{yprov4ml.get_artifacts_dir()}/results_visualization_0/outputs/{plot_acc_path}", 
                      is_output_to="results_visualization")

yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False, create_card=True)