
import os
import sys
import time
from typing import Any, Dict, Optional, Union
import prov.model as prov
import pwd
import warnings
import platform, threading, socket
from datetime import datetime

from yprov4ml.datamodel.artifact_data import ArtifactInfo
from yprov4ml.datamodel.metric_data import MetricInfo
from yprov4ml.datamodel.compressor_type import CompressorType, COMPRESSORS_FOR_ZARR
from yprov4ml.utils import funcs
from yprov4ml.utils.prov_utils import get_or_create_activity
from yprov4ml.utils.funcs import get_global_rank, get_runtime_type, get_local_timezone_info
from yprov4ml.utils.file_utils import _get_git_remote_url, _get_git_revision_hash, _get_source_files
from yprov4ml.utils.time_utils import get_time
from yprov4ml.utils.system_utils import Tracker

_RUN_ID_LOCK = threading.Lock()
_RUN_ID_CACHE = {}

def _new_run_id() -> int:
        return int(datetime.now().strftime("%Y%m%d%H%M%S"))

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
        self.PROV_PREFIX = "prov"
        self.XSD_PREFIX = "xsd"
        self.PROVML_PREFIX = "provml"
        self.yProv_PREFIX = "yprov4ml"
        self.RDF_PREFIX = "dcterms"

        self.RUN_ID = 0

        self.global_rank = None
        self.is_collecting = False

        self.save_metrics_after_n_logs = 100
        self.csv_separator = ","

        self.entities = {}
        self.contexts = {}
        self.context_timestamps = {}
        self.used_edges = set()
        self.gen_edges = set()

    def start_run(
            self, 
            experiment_name: str, 
            prov_save_path: Optional[str] = None, 
            user_namespace: Optional[str] = None, 
            collect_all_processes: bool = False, 
            save_after_n_logs: int = 100, 
            rank: Optional[int] = None, 
            metrics_file_type: str = "nc",
            csv_separator:str = ",", 
            use_compressor: Optional[Union[CompressorType, bool]] = None,
            use_run_id: Optional[int] = None, 
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

        if use_run_id is not None: 
            self.RUN_ID = use_run_id
        else:
            # if not os.path.exists(self.PROV_SAVE_PATH):
            #     os.makedirs(self.PROV_SAVE_PATH, exist_ok=True)
            #     self.RUN_ID = 0
            # else: 
            #     prev_exps = os.listdir(self.PROV_SAVE_PATH) 
            #     matching_files = [int(exp.split("_")[-1].split(".")[0]) for exp in prev_exps if funcs.prov4ml_experiment_matches(experiment_name, exp)]
            #     self.RUN_ID = max(matching_files)+1  if len(matching_files) > 0 else 0
            key = (self.PROV_SAVE_PATH, experiment_name)
            with _RUN_ID_LOCK:
                if use_run_id is not None:
                    self.RUN_ID = use_run_id
                else:
                    self.RUN_ID = _RUN_ID_CACHE.setdefault(key, _new_run_id())

        self.CLEAN_EXPERIMENT_NAME = experiment_name
        self.PROV_JSON_NAME = self.CLEAN_EXPERIMENT_NAME + f"_GR{self.global_rank}" if self.global_rank else experiment_name + f"_GR0"
        self.PROV_JSON_NAME = f"{self.PROV_JSON_NAME}_{self.RUN_ID}"

        self.EXPERIMENT_DIR = os.path.join(self.PROV_SAVE_PATH, f"{self.CLEAN_EXPERIMENT_NAME}_{self.RUN_ID}")
        self.ARTIFACTS_DIR = os.path.join(self.EXPERIMENT_DIR, f"artifacts_GR{self.global_rank}")
        self.METRIC_DIR = os.path.join(self.EXPERIMENT_DIR, f"metrics_GR{self.global_rank}")

        self.metrics_file_type = metrics_file_type
        self.use_compressor = use_compressor
        self.csv_separator = csv_separator
        self.tracker = Tracker()
        self.source_code_required = False

        self._init_provenance_doc()
        self._init_agent_context()

        # necessary when spawning threads, 
        # otherwise they get counted as different runs
        # TODO: find better approach
        time.sleep(1)
        os.makedirs(self.EXPERIMENT_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.METRIC_DIR, exist_ok=True)

    def _init_provenance_doc(self): 
        self.root_provenance_doc = prov.ProvDocument()
        self.root_provenance_doc.add_namespace(self.yProv_PREFIX, "https://hpci-lab.github.io/yProv4ML/ontology.html")
        self.root_provenance_doc.set_default_namespace(self.PROV_JSON_NAME)
        self.root_provenance_doc.add_namespace(self.PROV_PREFIX, "http://www.w3.org/ns/prov#")
        self.root_provenance_doc.add_namespace(self.XSD_PREFIX,'http://www.w3.org/2000/10/XMLSchema#')
        self.root_provenance_doc.add_namespace(self.PROVML_PREFIX, 'prov-ml')
        self.root_provenance_doc.add_namespace(self.RDF_PREFIX, 'http://purl.org/dc/terms/')

    def get_activity_attributes(self): 
        hostname = socket.gethostname()
        timezone = get_local_timezone_info()
        global_rank = get_global_rank()
        runtime_type = get_runtime_type()
        node_rank = os.getenv("SLURM_NODEID", None)
        local_rank = os.getenv("SLURM_LOCALID", None) 
        jobid = os.getenv("SLURM_JOB_ID", None) 
        jobname = os.getenv("SLURM_JOB_NAME", None) 

        d = {
            f"{self.yProv_PREFIX}:hostName": hostname,
            f"{self.yProv_PREFIX}:timeZone": timezone,
            f"{self.yProv_PREFIX}:runID": self.RUN_ID, 
            f"{self.yProv_PREFIX}:pythonVersion":str(sys.version), 
            f"{self.yProv_PREFIX}:runtimeType":str(runtime_type),
            f"{self.yProv_PREFIX}:operatingSystem":str(platform.system()),
            f"{self.yProv_PREFIX}:release":str(platform.release()),
            f"{self.yProv_PREFIX}:processor":str(platform.processor()),
            f"{self.yProv_PREFIX}:machine":str(platform.machine()),
        }

        if global_rank: d[f"{self.yProv_PREFIX}:globalRank"] = global_rank
        if local_rank: d[f"{self.yProv_PREFIX}:localRank"] = local_rank
        if node_rank: d[f"{self.yProv_PREFIX}:nodeRank"] = node_rank
        if jobid: d[f"{self.yProv_PREFIX}:jobID"] = jobid
        if jobname: d[f"{self.yProv_PREFIX}:jobName"] = jobname
        if global_rank: d[f"{self.yProv_PREFIX}:globalRank"] = global_rank

        return d



    def _init_agent_context(self): 
        user_ag = self.root_provenance_doc.agent(f'{pwd.getpwuid(os.getuid())[0]}')
        self.user_ag = user_ag
        user_ag.add_attributes({
            f"{self.yProv_PREFIX}:provenancePath":self.PROV_SAVE_PATH,
            f"{self.yProv_PREFIX}:artifactURI":self.ARTIFACTS_DIR,
            f"{self.yProv_PREFIX}:experimentDirectory":self.EXPERIMENT_DIR,
            f"{self.yProv_PREFIX}:experimentName":self.PROV_JSON_NAME,
            # f"{self.yProv_PREFIX}:PID":str(uuid.uuid4()), 
        })
        user_ag.add_attributes({f"{self.PROV_PREFIX}:startedAtTime": get_time()})

    def _add_ctx(self, ctx : str):
        if not ctx:
            return None
            
        if ctx in self.contexts:
            context = self.contexts[ctx]
            self.context_timestamps[ctx] = get_time()
            return context, False
        else:
            context, ctx_created = get_or_create_activity(self.root_provenance_doc, ctx)
            if ctx_created:         
                context.add_attributes({f"{self.PROV_PREFIX}:startedAtTime": str(get_time())})
                context.add_attributes(self.get_activity_attributes())

            self.contexts[ctx] = context
            self.context_timestamps[ctx] = get_time()
            context.wasAssociatedWith(self.user_ag)
            return context, True

    def _add_entity(self, path: str, attributes: dict):
        if path in self.entities:
            entity = self.entities[path]
            return entity, False
        else:
            entity = self.root_provenance_doc.entity(path, attributes)
            self.entities[path] = entity
            return entity, True

    def _create_nodes(self, path : str, is_input_to : Optional[str] = None, is_output_to: Optional[str]=None, attributes : dict={}): 
        entity, entity_created = self._add_entity(path, attributes)
        if entity_created: 
            entity.add_attributes({f'{self.PROV_PREFIX}:generatedAtTime': get_time()})

        entity_id = str(entity.identifier) if hasattr(entity, 'identifier') else str(path)

        if is_input_to: 
            is_input_to_obj, _ = self._add_ctx(is_input_to)
            ctx_id = str(is_input_to_obj.identifier) if hasattr(is_input_to_obj, 'identifier') else str(is_input_to)
            edge_key = (ctx_id, entity_id)
            if edge_key not in self.used_edges:
                is_input_to_obj.used(entity)
                self.used_edges.add(edge_key)

        if is_output_to: 
            is_output_to_obj, _ = self._add_ctx(is_output_to)
            ctx_id = str(is_output_to_obj.identifier) if hasattr(is_output_to_obj, 'identifier') else str(is_output_to)
            edge_key = (entity_id, ctx_id)
            if edge_key not in self.gen_edges:
                entity.wasGeneratedBy(is_output_to_obj)
                self.gen_edges.add(edge_key)

        role = None
        if is_input_to and not is_output_to: 
            role = "input"
        if is_output_to and not is_input_to: 
            role = "output"
        if role and entity_created: 
            entity.add_attributes({f"{self.PROV_PREFIX}:role": role})            
        return entity, is_output_to, is_output_to


    def finalize_activities(self):
        if not hasattr(self, 'contexts') or not hasattr(self, 'context_timestamps'):
            return

        for ctx_name, context in self.contexts.items():
            timestamps = self.context_timestamps.get(ctx_name, None)
            if timestamps:
                context.add_attributes({f"{self.PROV_PREFIX}:endedAtTime": str(timestamps)})

    def request_source_code(self, use_git : bool=False): 
        self.source_code_required = True
        self.use_git = use_git

    def add_source_code(self): 
        if self.use_git: 
            repo = _get_git_remote_url()
            if repo:
                commit_hash = _get_git_revision_hash()
                self.add_parameter("source_code", f"{repo}/{commit_hash}")
        else: 
            paths = _get_source_files()
            for path in paths: 
                os.makedirs(os.path.join(self.ARTIFACTS_DIR, "src"), exist_ok=True)
                self.add_artifact(path, path, is_model=False, is_input=True)

    def add_metric(self, metric: str, value: Any, step: int = 0, is_input_to: Optional[Any] = None, is_output_to: Optional[str] = None, timestamp: Optional[int] = None) -> None:
        timestamp = timestamp or funcs.get_current_time_millis()

        if (metric, is_output_to) not in self.metrics:
            self.metrics[(metric, is_output_to)] = MetricInfo(metric, is_input_to=is_input_to, is_output_to=is_output_to, use_compressor=self.use_compressor)
        
        self.metrics[(metric, is_output_to)].add_metric(value, step, timestamp)

        total_metrics_values = self.metrics[(metric, is_output_to)].total_metric_values
        if total_metrics_values % self.save_metrics_after_n_logs == 0:
            self.save_metric_to_file(self.metrics[(metric, is_output_to)])

    def add_parameter(self, parameter_name: str, parameter_value: Any, is_input_to : Optional[str] = None, is_output_to : Optional[str] = None) -> prov.ProvEntity:
        e, _, _ = self._create_nodes(parameter_name, is_input_to, is_output_to, {
            f"{self.PROV_PREFIX}:type": f"{self.PROVML_PREFIX}:HyperParameter", 
            f"{self.PROV_PREFIX}:label": parameter_name,
            f"{self.PROV_PREFIX}:value": str(parameter_value),
        })
        return e

    def add_artifact(
        self, 
        artifact_name: str, 
        artifact_path: str, 
        step: int = 0, 
        is_input_to : Optional[str] = None, 
        is_output_to : Optional[str] = None, 
        is_model : bool = False, 
        attributes : Optional[dict] = None, 
    ) -> prov.ProvEntity:

        e, is_input_to, is_output_to = self._create_nodes(artifact_name, is_input_to, is_output_to)

        self.artifacts[(artifact_name, is_input_to)] = ArtifactInfo(artifact_name, artifact_path, step, is_input_to=is_input_to, is_output_to=is_output_to, is_model=is_model)
        local_attrs = attributes.copy() if attributes else {}
        if f'{self.PROV_PREFIX}:label' not in local_attrs.keys(): 
            local_attrs[f'{self.PROV_PREFIX}:label'] = artifact_name
        if f'{self.RDF_PREFIX}:identifier' not in local_attrs.keys(): 
            local_attrs[f'{self.RDF_PREFIX}:identifier'] = artifact_path

        if is_model: 
            local_attrs.setdefault(f'{self.PROV_PREFIX}:type', f"{self.PROVML_PREFIX}:Model")

        if artifact_path: 
            file_size = os.path.getsize(artifact_path)
            local_attrs.setdefault(f'{self.yProv_PREFIX}:fileSizeInBytes', file_size)

        e.add_attributes(local_attrs)
        return e


    def save_metric_to_file(self, metric: MetricInfo) -> None:
        metric.save_to_file(self.METRIC_DIR, file_type=self.metrics_file_type, process=self.global_rank, csv_separator=self.csv_separator)

    def save_all_metrics(self) -> None:
        for metric in self.metrics.values():
            self.save_metric_to_file(metric)


