import torch.nn as nn
import zarr
import numpy as np
import threading
import queue
import atexit
from typing import Optional, Any
import shutil
from yprov4ml.constants import PROV4ML_DATA

ENCODING = np.float16
ENCODING_ZARR = "f2" if ENCODING == np.float16 else "f4"

# Stats saved per layer per epoch
STAT_KEYS = ["mean", "std", "min", "max", "p25", "p50", "p75"]


class ZarrWriterThread(threading.Thread):
    def __init__(self, zarr_wrapper):
        super().__init__(daemon=True)
        try:
            import torch
            torch.set_num_threads(1)
        except:
            pass
        self.wrapper = zarr_wrapper
        self.queue = queue.Queue(maxsize=256)
        self.start()

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            epoch, layer_stats = item
            self.wrapper._write_epoch_stats(epoch, layer_stats)


class WeightDistributionTrackedModel(nn.Module):
    def __init__(
        self,
        model_label: str,
        model: Any,
        context: Optional[str] = None,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_label
        self.model_path = f"{model_label}_weight_dist.zarr"
        self.context = context
        self.store = zarr.open(self.model_path, mode="w")
        self.chunk_size = chunk_size
        self.writer = ZarrWriterThread(self)
        self.writer_ptr = {}   # epoch pointer per (layer, stat)
        self.arrays = {}       # zarr arrays keyed by (layer, stat)
        self.initial_size = 32
        self._tracked_layers = self._collect_layers()
        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Layer collection
    # ------------------------------------------------------------------

    def _collect_layers(self) -> dict[str, nn.Module]:
        """Return {name: module} for all weight-bearing layers."""
        tracked = {}
        for name, module in self.model.named_modules():
            if isinstance(
                module,
                (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                 nn.Embedding, nn.LayerNorm, nn.BatchNorm1d,
                 nn.BatchNorm2d, nn.BatchNorm3d),
            ):
                if any(p.requires_grad for p in module.parameters(recurse=False)):
                    tracked[name] = module
        return tracked

    # ------------------------------------------------------------------
    # Zarr array management
    # ------------------------------------------------------------------

    def _ensure_arrays(self, layer_name: str):
        """Lazily create one 1-D resizable array per (layer, stat)."""
        if layer_name in self.arrays:
            return
        self.arrays[layer_name] = {}
        for stat in STAT_KEYS:
            path = f"{layer_name}/{stat}"
            
            if hasattr(self.store, "create_array"):
                # Zarr v3.0+ API
                arr = self.store.create_array(
                    path,
                    shape=(self.initial_size,),
                    chunks=(self.chunk_size,),
                    dtype=ENCODING_ZARR,
                    overwrite=True,
                    compressors=None,  # v3 uses plural 'compressors'
                )
            else:
                # Zarr v2.x legacy API
                arr = self.store.create_dataset(
                    path,
                    shape=(self.initial_size,),
                    chunks=(self.chunk_size,),
                    dtype=ENCODING_ZARR,
                    overwrite=True,
                    compressor=None,   # v2 uses singular 'compressor'
                )
                
            self.arrays[layer_name][stat] = arr
        self.writer_ptr[layer_name] = 0

    def _append_scalar(self, layer_name: str, stat: str, value: float):
        arr = self.arrays[layer_name][stat]
        ptr = self.writer_ptr[layer_name]   # same ptr for all stats of one layer
        if ptr >= arr.shape[0]:
            arr.resize((arr.shape[0] * 2,))
        arr[ptr] = value

    def _write_epoch_stats(self, epoch: int, layer_stats: dict):
        """Called from the writer thread."""
        for layer_name, stats in layer_stats.items():
            self._ensure_arrays(layer_name)
            for stat, value in stats.items():
                self._append_scalar(layer_name, stat, value)
            self.writer_ptr[layer_name] += 1   # advance after all stats written

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_epoch(self, epoch: int):
        """
        Call this at the end of every epoch (or every N steps).
        Computes weight statistics for every tracked layer and queues
        them for async writing.
        """
        layer_stats: dict[str, dict[str, float]] = {}

        for name, module in self._tracked_layers.items():
            # Flatten all *weight* parameters of this module into one tensor
            weight_tensors = [
                p.data.detach().float().cpu().flatten()
                for pname, p in module.named_parameters(recurse=False)
                if "weight" in pname
            ]
            if not weight_tensors:
                continue

            import torch
            weights = torch.cat(weight_tensors)
            q = torch.quantile(weights, torch.tensor([0.25, 0.50, 0.75]))

            layer_stats[name] = {
                "mean": float(weights.mean()),
                "std":  float(weights.std()),
                "min":  float(weights.min()),
                "max":  float(weights.max()),
                "p25":  float(q[0]),
                "p50":  float(q[1]),
                "p75":  float(q[2]),
            }

        self.writer.queue.put((epoch, layer_stats))

    def forward(self, x):
        return self.model(x)

    def close(self):
        # Drain the writer thread
        self.writer.queue.put(None)
        self.writer.join()

        PROV4ML_DATA.add_artifact(
            self.model_name,
            self.model_path,
            context=self.context,
            source="yProv4ML",
            is_input=False,
            log_copy_in_prov_directory=True,
            is_model=False,
        )
        shutil.rmtree(self.model_path)