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

class ZarrWriterThread(threading.Thread):
    def __init__(self, zarr_wrapper):
        super().__init__(daemon=True)

        try: 
            import torch
            torch.set_num_threads(1)
        except: pass

        self.wrapper = zarr_wrapper
        self.queue = queue.Queue(maxsize=256)
        self.start()

    def run(self):
        while True:
            items = []
            item = self.queue.get()
            if item is None:
                break

            items.append(item)

            while not self.queue.empty():
                try:
                    items.append(self.queue.get_nowait())
                except queue.Empty:
                    break

            # Now batch process
            for name, inp, out in items:
                inp = inp.to("cpu", non_blocking=True).numpy().astype(ENCODING)
                out = out.to("cpu", non_blocking=True).numpy().astype(ENCODING)
                self.wrapper._buffer_numpy(name, inp, out)

class ProvenanceTrackedModel(nn.Module):
    def __init__(self, model_label : str, model : Any, context : Optional[str] = None, chunk_size : int = 64):
        super().__init__()
        self.model = model
        self.model_name = model_label
        self.model_path = f"{model_label}.zarr"
        self.context = context

        self.store = zarr.open(self.model_path, mode='w')
        self.chunk_size = chunk_size
        self.handles = []
        self.layers = {}
        self.writer = ZarrWriterThread(self)
        self.writer_ptr = {}
        self.initial_size = 32

        self._register_hooks()

        atexit.register(self.close)

    def _ensure_arrays(self, name, inp, out):
        if name in self.layers:
            return

        in_shape = inp.shape[1:]
        out_shape = out.shape[1:]

        in_arr = self.store.create_dataset(
            f"{name}/input",
            shape=(self.initial_size, *in_shape),
            chunks=(self.chunk_size, *in_shape),
            dtype=ENCODING_ZARR,
            overwrite=True,
            compressor=None, 
        )

        out_arr = self.store.create_dataset(
            f"{name}/output",
            shape=(self.initial_size, *out_shape),
            chunks=(self.chunk_size, *out_shape),
            dtype=ENCODING_ZARR,
            overwrite=True,
            compressor=None, 
        )

        self.layers[name] = (in_arr, out_arr)
        self.writer_ptr[name] = 0

    def _buffer_numpy(self, name, inp, out):
        self._ensure_arrays(name, inp, out)
        in_arr, out_arr = self.layers[name]

        ptr = self.writer_ptr[name]

        self._append(in_arr, inp, ptr)
        self._append(out_arr, out, ptr)

        self.writer_ptr[name] += inp.shape[0]

    def _append(self, arr, data, ptr):
        M = arr.shape[0]
        new_n = ptr + data.shape[0]
        # arr.resize(M + data.shape[0], axis=0)
        if new_n >= M: 
            arr.resize((M*2, *arr.shape[1:]))
        arr[ptr:new_n] = data
        # arr[M:] = data

    def _hook_fn(self, name):
        def hook(module, inputs, output):
            inputs = inputs[0].detach()
            output = output.detach()
            self.writer.queue.put((name, inputs, output))
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def forward(self, x):
        return self.model(x)

    def close(self): 
        PROV4ML_DATA.add_artifact(self.model_name, self.model_path, context=self.context, source="yProv4ML", is_input=False, log_copy_in_prov_directory=True, is_model=False)
        shutil.rmtree(self.model_path)