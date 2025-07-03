
# Register Metrics for custom Operations

After collection of a specific metric, it's very often the case that a user may want to aggregate that information by applying functions such as mean, standard deviation, or min/max. 

yProv4ML allows to register a specific metric to be aggregated, using the function: 

```python
prov4ml.register_final_metric(
    metric_name : str,
    initial_value : float,
    fold_operation : FoldOperation
) 
```

where `fold_operation` indicates the function to be applied to the data. 

Several FoldOperations are already defined, such as MAX, MIN, ADD and SUBRACT. 
In any case the user is always able to define its own custom function, by either defining one with signature: 

```python
def custom_foldOperation(x, y): 
    return x // y
```

Or by passing a lambda function: 

```python
prov4ml.register_final_metric("my_metric", 0, lambda x, y: x // y) 
```

The output of the aggregated metric is saved in the PROV-JSON file, as an attribute of the current execution. 

[Home](README.md) | [Prev](time.md) | [Next](usage_pytorch.md)
