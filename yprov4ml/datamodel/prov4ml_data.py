
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
import prov.model as prov
import pwd
import warnings
import uuid

from yprov4ml.datamodel.artifact_data import ArtifactInfo
from yprov4ml.datamodel.metric_data import MetricInfo
from yprov4ml.datamodel.compressor_type import CompressorType, COMPRESSORS_FOR_ZARR
from yprov4ml.utils import funcs
from yprov4ml.utils.prov_utils import get_or_create_activity
from yprov4ml.utils.funcs import get_global_rank, get_runtime_type
from yprov4ml.utils.file_utils import _get_git_remote_url, _get_git_revision_hash, _get_source_files

class Prov4MLData:
    def __init__(self) -> None:
        self.metrics: Dict[(str, str), MetricInfo] = {}
        self.artifacts: Dict[(str, str), ArtifactInfo] = {}

        self.PROV_SAVE_PATH = "prov_save_path"
        self.PROV_JSON_NAME = "test_experiment"
        self.EXPERIMENT_DIR = "test_experiment_dir"
        self.ARTIFACTS_DIR = "artifact_dir"
        self.METRIC_DIR = "metric_dir"

        self.USER_NAMESPACE = "user_namespace"
        self.PROV_PREFIX = "yProv4ML_Entity"
        self.LABEL_PREFIX = "yProv4ML_Label"
        self.CONTEXT_PREFIX = "yProv4ML_Activity"
        self.SOURCE_PREFIX = "yProv4ML_Source"

        self.RUN_ID = 0

        self.global_rank = None
        self.is_collecting = False

        self.save_metrics_after_n_logs = 100
        self.csv_separator = ","

    def start_run(
            self, 
            experiment_name: str, 
            prov_save_path: Optional[str] = None, 
            user_namespace: Optional[str] = None, 
            collect_all_processes: bool = False, 
            save_after_n_logs: int = 100, 
            rank: Optional[int] = None, 
            disable_codecarbon : bool = False,
            metrics_file_type: str = "nc",
            csv_separator:str = ",", 
            use_compressor: Optional[Union[CompressorType, bool]] = None,
        ) -> None:

        self.global_rank = funcs.get_global_rank() if rank is None else rank
        self.is_collecting = self.global_rank is None or int(self.global_rank) == 0 or collect_all_processes
        
        if not self.is_collecting: 
            self.add_metric = lambda: None
            self.add_artifact = lambda: None
            self.save_metric_to_file = lambda: None
            self.save_all_metrics = lambda: None
            self.add_parameter = lambda: None
            return

        self.save_metrics_after_n_logs = save_after_n_logs
        if prov_save_path: self.PROV_SAVE_PATH = prov_save_path
        if user_namespace: self.USER_NAMESPACE = user_namespace

        if use_compressor in COMPRESSORS_FOR_ZARR and metrics_file_type != "zarr": 
            warnings.warn(f">start_run(): use_compressor chosen is only compatible with str.ZARR, but saving type is {metrics_file_type}, the compressor chosen will have no effect")
        if metrics_file_type == "zarr" and use_compressor != False and use_compressor not in COMPRESSORS_FOR_ZARR: 
            raise AttributeError(f">start_run(): use_compressor chosen is only compatible with str.ZARR")

        if metrics_file_type == "zarr" and use_compressor:
            use_compressor = CompressorType.BLOSC_ZSTD
        elif metrics_file_type in ["nc", "csv"] and use_compressor:
            use_compressor = CompressorType.ZIP
        if not use_compressor: 
            use_compressor = CompressorType.NONE

        # look at PROV dir how many experiments are there with the same name
        if not os.path.exists(self.PROV_SAVE_PATH):
            os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
            self.RUN_ID = 0
        else: 
            prev_exps = os.listdir(self.PROV_SAVE_PATH) 
            matching_files = [int(exp.split("_")[-1].split(".")[0]) for exp in prev_exps if funcs.prov4ml_experiment_matches(experiment_name, exp)]
            self.RUN_ID = max(matching_files)+1  if len(matching_files) > 0 else 0

        self.CLEAN_EXPERIMENT_NAME = experiment_name
        self.PROV_JSON_NAME = self.CLEAN_EXPERIMENT_NAME + f"_GR{self.global_rank}" if self.global_rank else experiment_name + f"_GR0"
        self.PROV_JSON_NAME = f"{self.PROV_JSON_NAME}_{self.RUN_ID}"

        self.EXPERIMENT_DIR = os.path.join(self.PROV_SAVE_PATH, f"{self.CLEAN_EXPERIMENT_NAME}_{self.RUN_ID}")
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, f"artifacts_GR{self.global_rank}")
        self.METRIC_DIR = os.path.join(self.EXPERIMENT_DIR, f"metrics_GR{self.global_rank}")

        self.metrics_file_type = metrics_file_type
        self.use_compressor = use_compressor
        self.csv_separator = csv_separator
        self.codecarbon_is_disabled = disable_codecarbon
        self.source_code_required = False

        self._init_root_context()

        # necessary when spawning threads, 
        # otherwise they get counted as different runs
        # TODO: find better approach
        time.sleep(1)
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.METRIC_DIR, exist_ok=True)

    def _add_ctx(self, rootstr : str, ctx : str, source : Optional[str] = None):
        rootstr = self._format_activity_name(rootstr)
        if source is not None: 
            src_context_name = self._format_activity_name(context=ctx, source=None)
            maybe_src_context, created = get_or_create_activity(self.root_provenance_doc, src_context_name)
            if created: 
                maybe_src_context.wasInformedBy(rootstr)

        context_name = self._format_activity_name(context=ctx, source=source)
        c, created = get_or_create_activity(self.root_provenance_doc, context_name)
        if created:         
            if source is not None: 
                c.wasInformedBy(maybe_src_context)
            else: 
                c.wasInformedBy(rootstr)
            # c.add_attributes({f'{self.LABEL_PREFIX}:level':1})
        return c

    def _set_ctx_or_default(self, ctx : str): 
        return ctx or self.PROV_JSON_NAME

    def _init_root_context(self): 
        self.root_provenance_doc = prov.ProvDocument()
        self.root_provenance_doc.add_namespace(self.CONTEXT_PREFIX, self.CONTEXT_PREFIX)
        self.root_provenance_doc.add_namespace(self.PROV_PREFIX, self.PROV_PREFIX)
        self.root_provenance_doc.add_namespace(self.LABEL_PREFIX, self.LABEL_PREFIX)
        self.root_provenance_doc.set_default_namespace(self.PROV_JSON_NAME)
        self.root_provenance_doc.add_namespace('prov','http://www.w3.org/ns/prov#')
        self.root_provenance_doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
        self.root_provenance_doc.add_namespace('prov-ml', 'prov-ml')

        user_ag = self.root_provenance_doc.agent(f'{pwd.getpwuid(os.getuid())[0]}')
        rootstr, _ = get_or_create_activity(self.root_provenance_doc, f"{self.CONTEXT_PREFIX}:{self.PROV_JSON_NAME}")
        rootstr.add_attributes({
            f'{self.LABEL_PREFIX}:level':0, 
            f"{self.LABEL_PREFIX}:provenance_path":self.PROV_SAVE_PATH,
            f"{self.LABEL_PREFIX}:artifact_uri":self.ARTIFACTS_DIR,
            f"{self.LABEL_PREFIX}:experiment_dir":self.EXPERIMENT_DIR,
            f"{self.LABEL_PREFIX}:experiment_name":self.PROV_JSON_NAME,
            f"{self.LABEL_PREFIX}:run_id":self.RUN_ID,
            f"{self.LABEL_PREFIX}:python_version":str(sys.version), 
            f"{self.LABEL_PREFIX}:PID":str(uuid.uuid4()), 
        })
        rootstr.wasAssociatedWith(user_ag)

        global_rank = get_global_rank()
        runtime_type = get_runtime_type()
        if runtime_type == "slurm":
            node_rank = os.getenv("SLURM_NODEID", None)
            local_rank = os.getenv("SLURM_LOCALID", None) 
            rootstr.add_attributes({
                f"{self.LABEL_PREFIX}:global_rank": str(global_rank),
                f"{self.LABEL_PREFIX}:local_rank":str(local_rank),
                f"{self.LABEL_PREFIX}:node_rank":str(node_rank),
            })
        elif runtime_type == "single_core":
            rootstr.add_attributes({
                f"{self.LABEL_PREFIX}:global_rank":str(global_rank)
            })

        self._add_ctx(self.PROV_JSON_NAME, self.PROV_JSON_NAME, 'std.time')

    def _format_activity_name(self, context : Optional[str] = None, source: Optional[str]=None): 
        context = self._set_ctx_or_default(context)
        return f"{self.CONTEXT_PREFIX}:{context}" + (f"-{self.SOURCE_PREFIX}:{source}" if source else "")

    def _format_artifact_name(self, label : str, context : Optional[str] = None, source: Optional[str]=None): 
        context = self._set_ctx_or_default(context)
        return f"{self.PROV_PREFIX}:{label}-{self.CONTEXT_PREFIX}:{context}" + (f"-{self.SOURCE_PREFIX}:{source}" if source else "")

    def _log_input(self, path : str, context : str, source: Optional[str]=None, attributes : dict={}) -> prov.ProvEntity:
        entity = self.root_provenance_doc.entity(path, attributes)
        # root_ctx = self._format_activity_name(self.PROV_JSON_NAME, None)
        activity = self._add_ctx(self.PROV_JSON_NAME, context, source)
        activity.used(entity)
        return entity
    
    def _log_output(self, path : str, context : str, source: Optional[str]=None, attributes : dict={}) -> prov.ProvEntity:
        entity= self.root_provenance_doc.entity(path, attributes)
        # root_ctx = self._format_activity_name(self.PROV_JSON_NAME, None)
        activity = self._add_ctx(self.PROV_JSON_NAME, context, source)
        entity.wasGeneratedBy(activity)
        return entity
    
    def request_source_code(self): 
        self.source_code_required = True

    def add_source_code(self): 
        repo = _get_git_remote_url()
        if repo is not None:
            commit_hash = _get_git_revision_hash()
            self.add_parameter(f"{self.LABEL_PREFIX}:source_code", f"{repo}/{commit_hash}")
        
        paths = _get_source_files()
        for path in paths: 
            os.makedirs(os.path.join(self.ARTIFACTS_DIR, "src"), exist_ok=True)
            self.add_artifact(path, path, log_copy_in_prov_directory=True, log_copy_subdirectory="src", is_model=False, is_input=True)

    def add_metric(self, metric: str, value: Any, step: int = 0, context: Optional[Any] = None, source: Optional[str] = None, timestamp: int = 0) -> None:
        context = self._set_ctx_or_default(context)

        if (metric, context) not in self.metrics:
            self.metrics[(metric, context)] = MetricInfo(metric, context, source=source, use_compressor=self.use_compressor)
        
        self.metrics[(metric, context)].add_metric(value, step, timestamp)

        total_metrics_values = self.metrics[(metric, context)].total_metric_values
        if total_metrics_values % self.save_metrics_after_n_logs == 0:
            self.save_metric_to_file(self.metrics[(metric, context)])

    def add_parameter(self, parameter_name: str, parameter_value: Any, context : Optional[str] = None, source : Optional[str] = None) -> None:
        context = self._set_ctx_or_default(context)

        root_ctx = self._format_activity_name(self.PROV_JSON_NAME, None)
        current_activity = self._add_ctx(root_ctx, context, source)
        current_activity.add_attributes({f"{self.LABEL_PREFIX}:{parameter_name}":str(parameter_value)})

    def _log_artifact_copy(self, artifact_path_src : str, artifact_path_dst : str, is_input : bool, is_model : bool, context : str, source : str): 
        try: 
            path = Path(artifact_path_src)
        except: 
            Exception(f">_log_artifact_copy: log_copy_in_prov_directory was True but value is not a valid Path: {artifact_path_src}, {artifact_path_dst}")
        
        newart_path = os.path.join(self.ARTIFACTS_DIR, artifact_path_dst)
        if path.is_file():
            # print(newart_path, "\n", os.path.dirname(newart_path))
            os.makedirs(os.path.dirname(newart_path), exist_ok=True)
            shutil.copy(path, newart_path)
        else:  
            shutil.copytree(path, newart_path)

        original = self.add_artifact("Original_" + path.name, str(path), log_copy_in_prov_directory=False, is_model=is_model, is_input=is_input, source=source, context=context)
        copied = self.add_artifact(path.name, newart_path, log_copy_in_prov_directory=False, is_model=is_model, is_input=is_input, source=source, context=context)
        copied.wasDerivedFrom(original)
        return copied


    def add_artifact(
        self, 
        artifact_name: str, 
        artifact_path: str, 
        step: int = 0, 
        context: Optional[Any] = None,
        source: Optional[str] = None,
        is_input : bool = False, 
        log_copy_in_prov_directory : bool = True, 
        log_copy_subdirectory : Optional[str] = None, 
        is_model : bool = False, 
    ) -> prov.ProvEntity:
        context = self._set_ctx_or_default(context)

        if log_copy_in_prov_directory: 
            new_path = os.path.join(log_copy_subdirectory, artifact_path) if log_copy_subdirectory else artifact_path
            return self._log_artifact_copy(artifact_path, new_path, is_input, is_model, context, source)

        artifact_name = self._format_artifact_name(artifact_name, context, source)
        self.artifacts[(artifact_name, context)] = ArtifactInfo(artifact_name, artifact_path, step, context=context, source=source, is_model=is_model)

        attributes = {
            f'{self.LABEL_PREFIX}:label': artifact_name, 
            f'{self.LABEL_PREFIX}:path': artifact_path,
        }

        if artifact_path: 
            file_size = os.path.getsize(artifact_path) / (1024*1024)
            attributes.setdefault(f'{self.LABEL_PREFIX}:file_size_in_mb', file_size)

        if is_input: 
            attributes.setdefault(f'{self.LABEL_PREFIX}:role','input')
            return self._log_input(artifact_name, context, source, attributes)
        else: 
            attributes.setdefault(f'{self.LABEL_PREFIX}:role', 'output')
            return self._log_output(artifact_name, context, source, attributes)

    def save_metric_to_file(self, metric: MetricInfo) -> None:
        metric.save_to_file(self.METRIC_DIR, file_type=self.metrics_file_type, process=self.global_rank, csv_separator=self.csv_separator)

    def save_all_metrics(self) -> None:
        for metric in self.metrics.values():
            self.save_metric_to_file(metric)


