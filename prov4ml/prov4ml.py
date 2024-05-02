import os
import mlflow
from mlflow import ActiveRun
from lightning.pytorch.loggers import MLFlowLogger
import prov.model as prov
import prov.dot as dot

from typing import Optional, Dict, Any
from contextlib import contextmanager

from .utils import energy_utils
from .utils import flops_utils
from .logging import log_execution_start_time, log_execution_end_time
from .provenance.provenance_graph import first_level_prov, second_level_prov

MLFLOW_SUBDIR = "mlflow"
ARTIFACTS_SUBDIR = "artifacts"
LIGHTNING_SUBDIR = "lightning"

@contextmanager
def start_run_ctx(
    prov_user_namespace: str,
    experiment_name: Optional[str] = None,
    provenance_save_dir: Optional[str] = None,
    mlflow_save_dir: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    log_system_metrics: Optional[bool] = None
    ) -> ActiveRun: # type: ignore
    """
    Starts an MLflow run and generates provenance information.

    Args:
        prov_user_namespace (str): The namespace of the user, this will be used as the default namespace.
        run_id (Optional[str]): The ID of the run to start. If not provided, a new run ID will be generated.
        experiment_id (Optional[str]): The ID of the experiment to associate the run with. If not provided, the default experiment will be used.
        run_name (Optional[str]): The name of the run. If not provided, a default name will be assigned.
        nested (bool): Whether the run is nested within another run. Defaults to False.
        tags (Optional[Dict[str, Any]]): Additional tags to associate with the run. Defaults to None.
        description (Optional[str]): A description of the run. Defaults to None.
        log_system_metrics (Optional[bool]): Whether to log system metrics. Defaults to None.

    Returns:
        ActiveRun: The active run object.

    Raises:
        None

    """
    global USER_NAMESPACE, PROV_SAVE_PATH, MLFLOW_SAVE_PATH, EXPERIMENT_NAME

    USER_NAMESPACE = prov_user_namespace
    PROV_SAVE_PATH = provenance_save_dir
    MLFLOW_SAVE_PATH = mlflow_save_dir
    EXPERIMENT_NAME = experiment_name

    if MLFLOW_SAVE_PATH:
        mlflow.set_tracking_uri(os.path.join(MLFLOW_SAVE_PATH, MLFLOW_SUBDIR))

    exp = mlflow.get_experiment_by_name(name=EXPERIMENT_NAME)
    if not exp:
        if MLFLOW_SAVE_PATH: 
            experiment_id = mlflow.create_experiment(
                name=EXPERIMENT_NAME,
                artifact_location=os.path.join(MLFLOW_SAVE_PATH, ARTIFACTS_SUBDIR)
            )
        else: 
            experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog(silent=True)

    current_run = mlflow.start_run(
        experiment_id=experiment_id,
        nested=nested,
        tags=tags,
        description=description,
        log_system_metrics=log_system_metrics, 
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

    yield current_run #return the mlflow context manager, same one as mlflow.start_run()

    log_execution_end_time()

    run_id=mlflow.active_run().info.run_id
    
    mlflow.end_run() #end the run, as per mlflow documentation

    client = mlflow.MlflowClient()
    active_run=client.get_run(run_id)

    doc = prov.ProvDocument()

    #set namespaces
    doc.set_default_namespace(USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('mlflow', 'mlflow') #TODO: find namespaces of mlflow and prov-ml ontologies
    doc.add_namespace('prov-ml', 'prov-ml')

    local_rank = os.getenv("SLURM_LOCALID", None)
    global_rank = os.getenv("SLURM_PROCID", None)
    node_id = os.getenv("SLURM_NODEID", None)

    doc = first_level_prov(active_run,doc, local_rank, global_rank, node_id)
    doc = second_level_prov(active_run,doc)    

    #datasets are associated with two sets of tags: input tags, of the DatasetInput object, and the tags of the dataset itself
    # for input_tag in dataset_input.tags:
    #     attributes[f'mlflow:{input_tag.key.strip("mlflow.")}']=str(input_tag.value)
    # for key,value in ds_tags['tags'].items():
    #     attributes[f'mlflow:{str(key).strip("mlflow.")}']=str(value)

    graph_filename = f'provgraph_{run_id}.json'
    dot_filename = f'provgraph_{run_id}.dot'
    path_graph = "/".join([PROV_SAVE_PATH, graph_filename]) if PROV_SAVE_PATH else graph_filename
    path_dot = "/".join([PROV_SAVE_PATH, dot_filename]) if PROV_SAVE_PATH else dot_filename

    if PROV_SAVE_PATH and not os.path.exists(PROV_SAVE_PATH):
        os.makedirs(PROV_SAVE_PATH)

    with open(path_graph,'w') as prov_graph:
        doc.serialize(prov_graph)
    with open(path_dot, 'w') as prov_dot:
        prov_dot.write(dot.prov_to_dot(doc).to_string())

def start_run(
    prov_user_namespace: str,
    experiment_name: Optional[str] = None,
    provenance_save_dir: Optional[str] = None,
    mlflow_save_dir: Optional[str] = None,
    nested: bool = False,
    tags: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    log_system_metrics: Optional[bool] = None
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
    global USER_NAMESPACE, PROV_SAVE_PATH, MLFLOW_SAVE_PATH, EXPERIMENT_NAME

    USER_NAMESPACE = prov_user_namespace
    PROV_SAVE_PATH = provenance_save_dir
    MLFLOW_SAVE_PATH = mlflow_save_dir
    EXPERIMENT_NAME = experiment_name

    if MLFLOW_SAVE_PATH:
        mlflow.set_tracking_uri(os.path.join(MLFLOW_SAVE_PATH, MLFLOW_SUBDIR))

    exp = mlflow.get_experiment_by_name(name=EXPERIMENT_NAME)
    if not exp:
        if MLFLOW_SAVE_PATH: 
            experiment_id = mlflow.create_experiment(
                name=EXPERIMENT_NAME,
                artifact_location=os.path.join(MLFLOW_SAVE_PATH, ARTIFACTS_SUBDIR)
            )
        else: 
            experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
    else:
        experiment_id = exp.experiment_id

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog(silent=True)

    mlflow.start_run(
        experiment_id=experiment_id,
        nested=nested,
        tags=tags,
        description=description,
        log_system_metrics=log_system_metrics, 
    )

    energy_utils._carbon_init()
    flops_utils._init_flops_counters()

    log_execution_start_time()

def end_run(): 
    """Ends the active MLflow run, generates provenance graph, and saves it."""
    
    log_execution_end_time()

    run_id=mlflow.active_run().info.run_id
    
    mlflow.end_run() #end the run, as per mlflow documentation

    client = mlflow.MlflowClient()
    active_run=client.get_run(run_id)

    doc = prov.ProvDocument()

    #set namespaces
    doc.set_default_namespace(USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('mlflow', 'mlflow') #TODO: find namespaces of mlflow and prov-ml ontologies
    doc.add_namespace('prov-ml', 'prov-ml')

    local_rank = os.getenv("SLURM_LOCALID", None)
    global_rank = os.getenv("SLURM_PROCID", None)
    node_id = os.getenv("SLURM_NODEID", None)

    doc = first_level_prov(active_run,doc, local_rank, global_rank, node_id)
    doc = second_level_prov(active_run,doc)    

    #datasets are associated with two sets of tags: input tags, of the DatasetInput object, and the tags of the dataset itself
    # for input_tag in dataset_input.tags:
    #     attributes[f'mlflow:{input_tag.key.strip("mlflow.")}']=str(input_tag.value)
    # for key,value in ds_tags['tags'].items():
    #     attributes[f'mlflow:{str(key).strip("mlflow.")}']=str(value)

    graph_filename = f'provgraph_{run_id}.json'
    dot_filename = f'provgraph_{run_id}.dot'
    path_graph = "/".join([PROV_SAVE_PATH, graph_filename]) if PROV_SAVE_PATH else graph_filename
    path_dot = "/".join([PROV_SAVE_PATH, dot_filename]) if PROV_SAVE_PATH else dot_filename

    if PROV_SAVE_PATH and not os.path.exists(PROV_SAVE_PATH):
        os.makedirs(PROV_SAVE_PATH)

    with open(path_graph,'w') as prov_graph:
        doc.serialize(prov_graph)
    with open(path_dot, 'w') as prov_dot:
        prov_dot.write(dot.prov_to_dot(doc).to_string())

def get_mlflow_logger():
    """Returns an MLFlowLogger instance configured with the specified parameters.

    Returns:
        MLFlowLogger: An MLFlowLogger instance.
    """ 
    mlf_logger = MLFlowLogger(
        experiment_name=EXPERIMENT_NAME,
        artifact_location=os.path.join(MLFLOW_SAVE_PATH, LIGHTNING_SUBDIR),
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=get_run_id(),
        log_model="all",
    )

    return mlf_logger

def get_run_id(): 
    """Returns the ID of the currently active MLflow run.

    Returns:
        str: The ID of the currently active MLflow run.
    """
    return mlflow.active_run().info.run_id
