import os
from argparse import Namespace
from typing import Any, Dict, Optional, Union

import yprov4ml
from lightning.pytorch import Callback
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor
from typing_extensions import override


def _unwrap(dl):
    """Lightning may wrap loaders in a CombinedLoader."""
    it = getattr(dl, "iterables", dl)
    if isinstance(it, (list, tuple)):
        return it[0]
    if isinstance(it, dict):
        return next(iter(it.values()))
    return it


class yProvMLLogger(Logger):
    """Logger that starts/ends the yprov4ml run and logs params/metrics
    with the right is_input_to / is_output_to context (e.g. Training_0).
    Pair it with ProvMLCallback to also log models and datasets."""

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        prov_user_namespace: str = "www.example.org",
        provenance_save_dir: str = "prov",
        save_after_n_logs: int = 100,
        collect_all_processes: bool = False,
        metrics_file_type: str = "csv",
        use_compressor: bool = False,
        create_graph: bool = True,
        create_svg: bool = True,
        create_ro_crate: bool = False,
        create_card: bool = True,
        log_system: bool = False,
        version: Optional[Union[int, str]] = None,
    ) -> None:
        super().__init__()
        self._name = experiment_name
        self._version = version
        self._save_dir = provenance_save_dir
        self._end_kwargs = dict(
            create_graph=create_graph,
            create_svg=create_svg,
            crate_ro_crate=create_ro_crate,  # (sic) argument name as in yprov4ml
            create_card=create_card,
        )
        self._log_system = log_system
        self._ended = False
        # context, updated by ProvMLCallback
        self.stage = "Training"
        self.epoch = 0

        yprov4ml.start_run(
            prov_user_namespace=prov_user_namespace,
            experiment_name=experiment_name,
            provenance_save_dir=provenance_save_dir,
            save_after_n_logs=save_after_n_logs,
            collect_all_processes=collect_all_processes,
            metrics_file_type=metrics_file_type,
            use_compressor=use_compressor,
        )

    @property
    def save_dir(self) -> str:
        return self._save_dir

    @property
    @override
    def root_dir(self) -> str:
        return os.path.join(self.save_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        v = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self.root_dir, v)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self) -> Optional[Union[int, str]]:
        return self._version

    @property
    def context(self) -> str:
        return f"{self.stage}_{self.epoch}"

    @override
    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Union[Tensor, float]], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            if k == "epoch":
                continue
            yprov4ml.log_metric(
                f"{k}_{self.epoch}", float(v), is_output_to=self.context, step=step or 0
            )
        if self._log_system:
            yprov4ml.log_system_metrics(is_output_to=self.context, step=step or 0)

    @override
    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args, **kwargs) -> None:
        if isinstance(params, Namespace):
            params = vars(params)
        for k, v in params.items():
            if not isinstance(v, (int, float, str, bool)):
                v = str(v)
            yprov4ml.log_param(k, v, is_input_to="Training_0")

    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        if not self._ended:
            self._ended = True
            yprov4ml.end_run(**self._end_kwargs)


class yProvMLCallback(Callback):
    """Reproduces the model / dataset / checkpoint provenance calls of the plain script."""

    def __init__(self, model_attrs: Optional[dict] = {}, dataset_attrs: Optional[dict] = {},
                 model_name: str = "parent"):
        self.model_attrs = model_attrs
        self.dataset_attrs = dataset_attrs
        self.model_name = model_name

    def _lg(self, trainer) -> Optional[yProvMLLogger]:
        for lg in trainer.loggers:
            if isinstance(lg, yProvMLLogger):
                return lg
        return None

    def _ckpt(self, epoch):
        return f"{yprov4ml.get_artifacts_dir()}/checkpoint_{epoch}"

    @rank_zero_only
    def on_fit_start(self, trainer, pl_module):
        if self._lg(trainer):
            yprov4ml.log_model(self.model_name, pl_module, is_input_to="Training_0", is_output_to="Download", other_attrs=self.model_attrs)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        lg = self._lg(trainer)
        if not lg:
            return
        lg.stage, lg.epoch = "Training", trainer.current_epoch
        e = lg.epoch
        if trainer.val_dataloaders is not None:
            yprov4ml.log_dataset("validation_data", _unwrap(trainer.val_dataloaders),
                                 is_input_to=f"Validation_{e}", is_output_to="Download",
                                 other_attrs=self.dataset_attrs)
        yprov4ml.log_dataset("training_data", _unwrap(trainer.train_dataloader),
                             is_input_to=f"Training_{e}", is_output_to="Download",
                             other_attrs=self.dataset_attrs)

    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        lg = self._lg(trainer)
        if lg and not trainer.sanity_checking:
            lg.stage = "Validation"
            lg.epoch = trainer.current_epoch

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        lg = self._lg(trainer)
        if lg and not trainer.sanity_checking:
            yprov4ml.log_model(self._ckpt(lg.epoch), pl_module, is_input_to=f"Validation_{lg.epoch}")
            lg.stage = "Training"

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        lg = self._lg(trainer)
        if not lg:
            return
        e = trainer.current_epoch
        if e + 1 < trainer.max_epochs:
            yprov4ml.log_model(self._ckpt(e), pl_module, is_input_to=f"Training_{e+1}", is_output_to=f"Training_{e}")
        else:
            yprov4ml.log_model(self._ckpt(e), pl_module, is_output_to=f"Training_{e}")

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        lg = self._lg(trainer)
        if lg:
            yprov4ml.log_model(f"{yprov4ml.get_artifacts_dir()}/weights", pl_module, is_output_to=f"Training_{trainer.current_epoch - 1}")