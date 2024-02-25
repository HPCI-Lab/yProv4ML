import torch
import mlflow
from timeit import default_timer as timer
#from context_manager import log_metrics,Context
import prov4ml.prov4ml as prov4ml

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
    """Performs a single training step
    Args:
        model: A torch.nn.Module model to be trained
        dataloader: A torch.utils.data.DataLoader for training data
        loss_fn: a torch.nn.Module loss function to calculate loss
        optimizer: a torch.optim.Optimizer used for optimization
    Returns
        A tuple (train_loss, train_accuracy) representing loss and accuracy measures on the whole dataset
    """
    model.train()
    running_loss, running_corrects,running_total = 0.0, 0, 0

    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*X.size(0)
        _,predicted = torch.max(y_pred.data,1)
        running_total += y.size(0)
        running_corrects += (predicted==y).sum().item() 
    train_loss = running_loss / len(dataloader.dataset)
    train_acc = running_corrects / running_total
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module):
    """Performs a single evaluation step
    Args:
        model: a torch.nn.Module model to be evaluated
        dataloader: a torch.utils.data.DataLoader of the data used for validation
        loss_fn: a torch.nn.Module loss function used to calculate loss
    Returns:
        A tuple (test_loss,test_acc) representing loss and accuracy measures on the whole dataset. It also returns the last prediction logits
    """
    model.eval() 
    running_loss, running_corrects, running_total = 0.0, 0, 0
    with torch.inference_mode():
        test_pred_logits= 0
        for batch, (X, y) in enumerate(dataloader):
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)

            _,predicted = torch.max(test_pred_logits.data,1)
            running_total += y.size(0)
            running_corrects += (predicted==y).sum().item()
            running_loss += loss.item()*y.size(0)
    test_loss= running_loss/ len(dataloader.dataset)
    test_acc = running_corrects/running_total
    return test_loss, test_acc, test_pred_logits

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn:torch.nn.Module,
          epochs: int):
    """Function performing the training of the model
    Args:
        model: a torch.nn.Module model to be trained
        train_dataloader: a torch.utils.data.DataLoader of the training dataset
        test_dataloader: a torch.utils.data.DataLoader of the evaluation dataset
        optimizer: a torch.optim.Optimizer used for optimization
        loss_fn: a torch.nn.Module loss function for loss calculation
        epochs: an int for the training epochs
    Returns:
        a dictionary {
            "train_loss":[],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        containing the results of the training process
    """
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in range(epochs):
        train_start=timer()
        train_loss, train_acc = train_step(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
        train_end=timer()
        test_loss, test_acc, pred_logits = test_step(model=model,dataloader=test_dataloader,loss_fn=loss_fn)
        test_end=timer()
        # mlflow.log_metrics({
        #     "train_loss":train_loss,
        #     "train_acc":train_acc,
        #     "test_loss":test_loss,
        #     "test_acc":test_acc
        # },step=epoch)
        prov4ml.log_metrics({
            "train_loss":(train_loss,prov4ml.Context.TRAINING),
            "train_acc":(train_acc,prov4ml.Context.TRAINING),
            "test_loss":(test_loss,prov4ml.Context.EVALUATION),
            "test_acc":(test_acc,prov4ml.Context.EVALUATION),
            "train_time":(train_end-train_start,prov4ml.Context.TRAINING),
            "test_time":(test_end-train_end,prov4ml.Context.EVALUATION)
        },step=epoch)
        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        state_dict ={
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss_fn
        }
        
        mlflow.pytorch.log_state_dict(state_dict,artifact_path=f"checkpoint/{epoch}")
        mlflow.log_text(str(pred_logits),f"pred_logits/{epoch}.txt")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results