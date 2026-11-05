# Reproducible Example

<div style="display: flex; align-items: center; background-color: #cc3300; color: #333; border: 5px solid #cc3300; font-weight: bold; border-radius: 5px; position: relative;">
    <span style="position: absolute; left: 10px; font-size: 20px;">‼</span>
    <span style="margin-left: 35px; padding: 5px; background-color: white; border-radius: 5px; width: 100%">This feature is still under development. Currently at the [yProv4REP](https://github.com/HPCI-Lab/yProv4Rep) Repository </span>
</div>


<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
import yprov4ml

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="reproducible_example", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=True, 
)

yprov4ml.log_source_code()
yprov4ml.log_execution_command()

def square(x): 
    return x**2

for i in range(1, 10): 
    yprov4ml.log_param(f"input_{i}", i, is_input_to="square")
    o = square(i)
    yprov4ml.log_param(f"output_{i}", o, is_output_to="square")

yprov4ml.end_run(create_graph=True, create_svg=True)
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">



<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="usage_itwinAI_logger.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>