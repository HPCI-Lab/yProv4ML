import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
sys.path.append("yprov4ml")
import yprov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 8
EPOCHS = 3
DEVICE = "cpu"

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="cellpose-sam-finetuned-hpa", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=False, 
    metrics_file_type="csv",
    use_compressor=False, 
)

yprov4ml.log_param("C", "value3", is_input_to="1")
yprov4ml.log_param("A", "value", is_input_to="1", is_output_to="2")
yprov4ml.log_param("B", "value2", is_input_to="2", is_output_to="3")




yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False, create_card=True)
