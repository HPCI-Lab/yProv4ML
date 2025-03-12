import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTFeatureExtractor
import sys
sys.path.append("../ProvML")
import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cpu"

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="transformer_finetuning",
    provenance_save_dir="prov",
    save_after_n_logs=1,
)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

def transform_fn(image):
    return feature_extractor(images=image, return_tensors="pt")['pixel_values'][0]

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: transform_fn(x))
])

train_ds = CIFAR100(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(2000))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = CIFAR100(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(test_loader, "val_dataset")

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=100).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=5e-5)
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
prov4ml.log_param("loss_fn", "CrossEntropyLoss")

losses = []
for epoch in range(EPOCHS):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        outputs = model(x).logits
        loss = loss_fn(outputs, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        
        prov4ml.log_metric("Loss", loss.item(), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)

    model.eval()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(test_loader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x).logits
            loss = loss_fn(outputs, y)
            prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.VALIDATION, step=epoch)

prov4ml.log_model(model, "vit_cifar100_final")
prov4ml.end_run(create_graph=True, create_svg=True)
