
# Provenance Collection Creation

The provenance collection functionality can be used to create a summary file linking all PROV-JSON files generated during a run. These files come from distributed execution, where each process generates its own log file, and the user may want to create a single file containing all the information.

The collection can be created with the following command: 

```bash
python -m prov4ml.prov_collection --experiment_path experiment_path --output_dir output_dir
```

Where `experiment_path` is the path to the experiment directory containing all the PROV-JSON files, and `output_dir` is the directory where the collection file will be saved. 

[Home](README.md) | [Prev](prov_collection.md) | [Next](carbon.md)
