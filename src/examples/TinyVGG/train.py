#!/usr/bin/env python3

import torch
from torch import nn
from torchvision import transforms
from torchinfo import summary
import numpy as np
import mlflow

import argparse
import os

from data_setup import load_dataset
from model_builder import TinyVGG
from engine import train
#from context_manager import start_run
import prov4ml.prov4ml as prov4ml
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-E","--experiment_name",help="Name to be assigned to the experiment")
    parser.add_argument("-R","--run_name", help="Name given to the run")
    parser.add_argument("--data_dir", help="Path where the dataset will be saved", type=os.PathLike)
    parser.add_argument("--batch_size",help="Dataloader batch size", type=int)
    parser.add_argument("-N","--num_epochs", help="Number of training epochs", type=int)
    parser.add_argument("--lr", help="Learning rate for the optimizer",type=float)
    parser.add_argument("--hidden_units",help="Number of hidden units in the model", type=int)

    args = parser.parse_args()
    EXPERIMENT_NAME=args.experiment_name if args.experiment_name is not None else "TinyVGG2"
    RUN_NAME=args.run_name if args.run_name is not None else 'run'
    DATA_DIR=args.data_dir if args.data_dir is not None else './data'
    BATCH_SIZE=args.batch_size if args.batch_size is not None else 32
    NUM_EPOCHS=args.num_epochs if args.num_epochs is not None else 20
    LR=args.lr if args.lr is not None else 0.1
    HIDDEN_UNITS=args.hidden_units if args.hidden_units is not None else 10

    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)
    
    with prov4ml.start_run(prov_user_namespace="www.example.org",run_name=RUN_NAME):
        
        train_transform = transforms.Compose([transforms.Resize(size=(28,28)),transforms.TrivialAugmentWide(num_magnitude_bins=31),transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.Resize((28,28)),transforms.ToTensor()])
        train_dataloader, test_dataloader,class_names = load_dataset(DATA_DIR,train_transform,test_transform,BATCH_SIZE)


        #training data samples
        X = next(iter(train_dataloader))[0].numpy()
        y= next(iter(train_dataloader))[1].numpy()

        model = TinyVGG(input_shape=X.shape[1], # number of color channels (3 for RGB) 
                        hidden_units=HIDDEN_UNITS,
                        output_shape=len(class_names))


        
        summary(model,input_size=X.shape)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)

        mlflow.log_params({
            "epochs":NUM_EPOCHS,
            "hidden_units":HIDDEN_UNITS,
            "loss_fn":loss_fn._get_name(),
            "optimizer":optimizer.__class__.__name__,
            "lr":LR
        })
        model_0_results = train(model,train_dataloader,test_dataloader,optimizer,loss_fn,NUM_EPOCHS)
        
    
        signature = mlflow.models.infer_signature(X,y)
        mlflow.pytorch.log_model(pytorch_model=model,artifact_path=model._get_name(),registered_model_name=model._get_name(),signature=signature)

if __name__ == '__main__':
    main()