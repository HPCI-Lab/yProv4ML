

# Execution Time

```python
yprov4ml.log_current_execution_time(
    label: str, 
    is_input_to: Optional[str] = None, 
    is_output_to: Optional[str] = 'std.time', 
    step: Optional[int] = None
)
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `label` | `string` | **Required**. Label of the code portion |
| `is_input_to` | `Optional[str]` | **Optional**. Activity that consumes the metric |
| `is_output_to` | `Optional[str]` | **Optional**. Producer of the metric (default `std.time`) |
| `step` | `int` | **Optional**. Step of the metric |

<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="system.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="registering_metrics.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>