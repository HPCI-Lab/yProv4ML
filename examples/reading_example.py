
import sys

from yprov4ml.utils import getters

if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else "yProv4ML/prov/example_4/prov_example_GR0_4.json"
    # 1. list_activities
    print("=== Activities ===")
    for a in getters.list_activities(path):
        print(" ", a)
    print()

    # 2. list_entities (all, then filtered)
    print("=== Entities (models only) ===")
    for e in getters.list_entities(path, "provml:Model"):
        print(" ", e)
    print()

    # 3. get_parameter
    print("=== Parameters ===")
    print("run_id          :", getters.get_parameter(path, "example_GR0_4", "yprov:run_id"))
    print("runtime_type    :", getters.get_parameter(path, "example_GR0_4", "yprov:runtime_type"))
    print("batch_size      :", getters.get_parameter(path, "train_dataset//Training", "yprov:train_dataset_stat_batch_size"))
    print("Training start  :", getters.get_parameter(path, "Training", "prov:startedAtTime"))
    print()

    print("All parameters: ", getters.list_parameters(path, "train_dataset//Training"))
    print()
    print("paths: ", getters.list_metric_paths(path, context='Validation'))
    print()

    # 4. get_metrics_as_df
    print("=== Metrics DataFrame (Training) ===")
    df = getters.list_metrics(path, context="Training")
    print(df[["label", "context", "source", "yprov:file_size"]].to_string())
    print()
