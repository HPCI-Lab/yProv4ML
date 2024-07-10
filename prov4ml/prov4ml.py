import os
import prov.dot as dot

from typing import Optional
from contextlib import contextmanager

from .constants import PROV4ML_DATA
from .utils import energy_utils
from .utils import flops_utils
from .logging import log_execution_start_time, log_execution_end_time
from .provenance.provenance_graph import create_prov_document
from .datamodel.prov4ml_collection import create_prov_collection

@contextmanager
def start_run_ctx(
    prov_user_namespace: str,
    experiment_name: Optional[str] = None,
    provenance_save_dir: Optional[str] = None,
    collect_all_processes: Optional[bool] = False,
    save_after_n_logs: Optional[int] = 100,
    create_graph: Optional[bool] = False, 
    create_svg: Optional[bool] = False,
    ) -> None: # type: ignore
    """
    Starts an MLflow run and generates provenance information.

    Args:
        prov_user_namespace (str): The user namespace for provenance tracking.
        experiment_name (Optional[str], optional): The name of the experiment to associate the run with. Defaults to None.
        provenance_save_dir (Optional[str], optional): The directory to save provenance data. Defaults to None.
        mlflow_save_dir (Optional[str], optional): The directory to save MLflow artifacts. Defaults to None.
        nested (bool, optional): If True, starts a nested run. Defaults to False.
        tags (Optional[Dict[str, Any]], optional): Dictionary of tags to associate with the run. Defaults to None.
        description (Optional[str], optional): Description of the run. Defaults to None.
        log_system_metrics (Optional[bool], optional): If True, logs system metrics during the run. Defaults to None.

    Returns:
        ActiveRun: The active run object.

    """
    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")

    PROV4ML_DATA.init(
        experiment_name=experiment_name, 
        prov_save_path=provenance_save_dir, 
        user_namespace=prov_user_namespace, 
        collect_all_processes=collect_all_processes, 
        save_after_n_logs=save_after_n_logs
    )
   
    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

    yield None#current_run #return the mlflow context manager, same one as mlflow.start_run()

    log_execution_end_time()

    doc = create_prov_document()

    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'
    dot_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.dot'

    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)

    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)

    with open(path_graph,'w') as prov_graph:
        doc.serialize(prov_graph)

    if create_graph:
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(dot.prov_to_dot(doc).to_string())

    create_prov_collection()

def start_run(
    prov_user_namespace: str,
    experiment_name: Optional[str] = None,
    provenance_save_dir: Optional[str] = None,
    collect_all_processes: Optional[bool] = False,
    save_after_n_logs: Optional[int] = 100,
    ) -> None:
    """Starts an MLflow run with the specified configurations and options.
    
    Args:
        prov_user_namespace (str): The user namespace for provenance tracking.
        experiment_name (Optional[str], optional): The name of the experiment to associate the run with. Defaults to None.
        provenance_save_dir (Optional[str], optional): The directory to save provenance data. Defaults to None.
        mlflow_save_dir (Optional[str], optional): The directory to save MLflow artifacts. Defaults to None.
        nested (bool, optional): If True, starts a nested run. Defaults to False.
        tags (Optional[Dict[str, Any]], optional): Dictionary of tags to associate with the run. Defaults to None.
        description (Optional[str], optional): Description of the run. Defaults to None.
        log_system_metrics (Optional[bool], optional): If True, logs system metrics during the run. Defaults to None.
    """

    PROV4ML_DATA.init(
        experiment_name=experiment_name, 
        prov_save_path=provenance_save_dir, 
        user_namespace=prov_user_namespace, 
        collect_all_processes=collect_all_processes, 
        save_after_n_logs=save_after_n_logs
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(
        create_graph: Optional[bool] = False, 
        create_svg: Optional[bool] = False
        ): 
    """Ends the active MLflow run, generates provenance graph, and saves it."""

    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")
    
    if not PROV4ML_DATA.is_collecting: return
    
    log_execution_end_time()

    # save remaining metrics
    PROV4ML_DATA.save_all_metrics()

    doc = create_prov_document()
   
    graph_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.json'
    dot_filename = f'provgraph_{PROV4ML_DATA.EXPERIMENT_NAME}.dot'
    
    if not os.path.exists(PROV4ML_DATA.EXPERIMENT_DIR):
        os.makedirs(PROV4ML_DATA.EXPERIMENT_DIR, exist_ok=True)
    
    path_graph = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, graph_filename)

    with open(path_graph,'w') as prov_graph:
        doc.serialize(prov_graph)

    if create_graph:
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(dot.prov_to_dot(doc).to_string())

    if create_svg:
        os.system(f"dot -Tsvg -O {path_dot}")

    create_prov_collection()

