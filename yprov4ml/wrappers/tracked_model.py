import torch.nn as nn
import zarr
import numpy as np
import threading
import queue

class ZarrWriterThread(threading.Thread):
    def __init__(self, zarr_wrapper):
        super().__init__(daemon=True)

        try: 
            import torch
            torch.set_num_threads(1)
        except: pass

        self.wrapper = zarr_wrapper
        self.queue = queue.Queue(maxsize=1024)
        self.start()

    def run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break

            name, inp, out = item
            inp = inp.cpu().numpy().astype(np.float32)
            out = out.cpu().numpy().astype(np.float32)
            self.wrapper._buffer_numpy(name, inp, out)

class ProvenanceTrackedModel(nn.Module):
    def __init__(self, model, store_path="provenance.zarr", chunk_size=256):
        super().__init__()
        self.model = model
        self.store = zarr.open(store_path, mode='w')
        self.chunk_size = chunk_size
        self.handles = []
        self.layers = {}
        self.writer = ZarrWriterThread(self)
        self.writer_ptr = {}
        self.initial_size = 1000

        self._register_hooks()

    def _ensure_arrays(self, name, inp, out):
        if name in self.layers:
            return

        in_shape = inp.shape[1:]
        out_shape = out.shape[1:]

        in_arr = self.store.create_dataset(
            f"{name}/input",
            shape=(self.initial_size, *in_shape),
            chunks=(self.chunk_size, *in_shape),
            dtype='f4',
            overwrite=True,
        )

        out_arr = self.store.create_dataset(
            f"{name}/output",
            shape=(self.initial_size, *out_shape),
            chunks=(self.chunk_size, *out_shape),
            dtype='f4',
            overwrite=True,
        )

        self.layers[name] = (in_arr, out_arr)
        self.writer_ptr[name] = 0

    def _buffer_numpy(self, name, inp, out):
        self._ensure_arrays(name, inp, out)
        in_arr, out_arr = self.layers[name]

        ptr = self.writer_ptr[name]

        self._append(in_arr, inp, ptr)
        self._append(out_arr, out, ptr)

    def _append(self, arr, data, ptr):
        M = arr.shape[0]
        new_n = ptr + data.shape[0]
        if new_n >= M: 
            arr.resize((M*2, *arr.shape[1:]))
        arr[ptr:new_n] = data

    def _hook_fn(self, name):
        def hook(module, inputs, output):
            inp = inputs[0].detach()
            out = output.detach()

            self.writer.queue.put((name, inp, out))
        return hook

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(self._hook_fn(name))
                self.handles.append(handle)

    def forward(self, x):
        return self.model(x)

    def close(self):
        for h in self.handles:
            h.remove()
