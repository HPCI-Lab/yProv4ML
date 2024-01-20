import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

import mlflow

def load_dataset(data_dir:str, train_transform:transforms.Compose, test_transform:transforms.Compose, batch_size:int):
    """Loads the FashionMNIST dataset and returns the torchvision Dataloaders
    Args:
        data_dir: path to data directory
        train_transform: torchvision transforms to perform on training data
        test_transform: torchvision transforms to perform on validaton data
        batch_size: number of samples per batch
    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)    
    """
    #train_transform = transforms.Compose([transforms.Resize(size=(28,28)),transforms.TrivialAugmentWide(num_magnitude_bins=31),transforms.ToTensor()])
    #test_transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
    
    train_data = datasets.FashionMNIST(root=data_dir,train=True,transform=train_transform,download=True)
    test_data = datasets.FashionMNIST(root=data_dir,train=False,transform=test_transform, download=True)
    class_names = train_data.classes

    train_dataloader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    test_dataloader= DataLoader(test_data, batch_size=batch_size,shuffle=False)
    
    train_numpy = next(iter(train_dataloader))[0].numpy()
    test_numpy = next(iter(test_dataloader))[0].numpy()
    labels_numpy = next(iter(train_dataloader))[1].numpy()

    mlflow.log_input(mlflow.data.numpy_dataset.from_numpy(train_numpy),context="training",tags={'source_mirror':f'{train_data.mirrors[0]}','source_resources':f'{train_data.resources}','transforms':str(train_transform.transforms)})
    mlflow.log_input(mlflow.data.numpy_dataset.from_numpy(test_numpy), context="testing",tags={'source_mirror':f'{test_data.mirrors[0]}','source_resources':f'{test_data.resources}','transforms':str(test_transform.transforms)})


    return train_dataloader, test_dataloader, class_names