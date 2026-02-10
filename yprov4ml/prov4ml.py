import os
from typing import Optional, Union

from yprov4ml.constants import PROV4ML_DATA
from yprov4ml.utils import energy_utils
from yprov4ml.utils import flops_utils
from yprov4ml.logging_aux import log_execution_start_time, log_execution_end_time
from yprov4ml.provenance.provenance_graph import create_prov_document, create_rocrate_in_dir, save_prov_file
from yprov4ml.utils.file_utils import _requirements_lookup
from yprov4ml.datamodel.compressor_type import CompressorType

def start_run(
        experiment_name: str,
        prov_user_namespace: Optional[str] = None,
        provenance_save_dir: Optional[str] = None,
        collect_all_processes: Optional[bool] = False,
        save_after_n_logs: Optional[int] = 100,
        rank : Optional[int] = None, 
        disable_codecarbon : Optional[bool] = False,
        metrics_file_type: str = "csv",
        csv_separator : str = ",", 
        use_compressor: Optional[Union[CompressorType, bool]] = None,
    ) -> None:
    PROV4ML_DATA.start_run(
        experiment_name=experiment_name, 
        prov_save_path=provenance_save_dir, 
        user_namespace=prov_user_namespace, 
        collect_all_processes=collect_all_processes, 
        save_after_n_logs=save_after_n_logs, 
        rank=rank, 
        disable_codecarbon=disable_codecarbon, 
        metrics_file_type=metrics_file_type,
        csv_separator=csv_separator,
        use_compressor=use_compressor,
    )

    if not disable_codecarbon: 
        energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(create_graph: Optional[bool] = False, create_svg: Optional[bool] = False, crate_ro_crate: Optional[bool]=False):  
    if not PROV4ML_DATA.is_collecting: return
    
    log_execution_end_time()

    filename = _requirements_lookup("./")
    PROV4ML_DATA.add_artifact("requirements", filename, step=0, context=None, is_input=True)
    

    if PROV4ML_DATA.source_code_required: 
        PROV4ML_DATA.add_source_code()

    PROV4ML_DATA.save_all_metrics()

    doc = create_prov_document()
   
    graph_filename = f'prov_{PROV4ML_DATA.PROV_JSON_NAME}.json'
    
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)
    
    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)
    save_prov_file(doc, path_graph, create_graph, create_svg)

    if crate_ro_crate: 
        create_rocrate_in_dir(PROV4ML_DATA.EXPERIMENT_DIR)
