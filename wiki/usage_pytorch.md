
# Example of usage with PyTorch

This section provides an example of how to use Prov4ML with PyTorch.

The following code snippet shows how to log metrics, system metrics, carbon metrics, and model versions in a PyTorch training loop.

#### Example

```python
for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = mnist_model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optim.step()
        prov4ml.log_metric("MSE_train", loss, context=prov4ml.Context.TRAINING, step=epoch)
    
    # log system and carbon metrics (once per epoch), as well as the execution time
    prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
    prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)
    # save incremental model versions
    prov4ml.save_model_version(mnist_model, f"mnist_model_version_{epoch}", prov4ml.Context.TRAINING, epoch)
     
```


[Home](README.md) | [Prev](registering_metrics.md) | [Next](usage_lightning.md)
