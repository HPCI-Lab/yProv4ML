
import os
from typing import Any, Dict, List
from typing import Optional
import zarr
import netCDF4 as nc

from yprov4ml.datamodel.compressor_type import compressor_to_type

ZARR_CHUNK_SIZE = 1000

class MetricInfo:
    __slots__ = ['name', 'context', 'source', 'total_metric_values', 'use_compressor', 'epochDataList']
    
    def __init__(self, name: str, context: Any, source=str, use_compressor : Optional[str] = None) -> None:
        self.name = name
        self.context = context
        self.source = source
        self.total_metric_values = 0
        self.use_compressor = use_compressor
        self.epochDataList: Dict[int, List[Any]] = {}

    def add_metric(self, value: Any, epoch: int, timestamp : int) -> None:
        if epoch not in self.epochDataList:
            self.epochDataList[epoch] = []

        self.epochDataList[epoch].append((value, timestamp))
        self.total_metric_values += 1

    def save_to_file(
            self, 
            path: str, 
            file_type: str,
            csv_separator : str = ",", 
            process: Optional[int] = None, 
        ) -> None:

        process = process if process is not None else 0
        file = os.path.join(path, f"{self.name}_{self.context}_{self.source}_GR{process}")

        ft = f"{file}.{file_type}"
        if file_type == "zarr":
            self.save_to_zarr(ft)
        elif file_type == "csv":
            self.save_to_txt(ft, csv_separator)
        elif file_type == "nc":
            self.save_to_netCDF(ft)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        self.epochDataList = {}

    def save_to_netCDF(self, netcdf_file: str) -> None:

        if os.path.exists(netcdf_file):
            dataset = nc.Dataset(netcdf_file, mode='a', format='NETCDF4')
        else:
            dataset = nc.Dataset(netcdf_file, mode='w', format='NETCDF4')
            dataset._name = self.name
            dataset._context = str(self.context)
            dataset._source = str(self.source)

            compression = 'zlib' if self.use_compressor else None
            dataset.createDimension('time', None)
            dataset.createVariable('epochs', 'i4', ('time',), compression)
            dataset.createVariable('values', 'f4', ('time',), compression)
            dataset.createVariable('timestamps', 'i8', ('time',), compression)

        epochs = []
        values = []
        timestamps = []

        for epoch, items in self.epochDataList.items():
            for value, timestamp in items:
                epochs.append(epoch)
                values.append(value)
                timestamps.append(timestamp)

        current_size = dataset.dimensions['time'].size
        new_size = current_size + len(epochs)

        dataset.variables['epochs'][current_size:new_size] = epochs
        dataset.variables['values'][current_size:new_size] = values
        dataset.variables['timestamps'][current_size:new_size] = timestamps

        dataset.close()

    def save_to_zarr(self, zarr_file: str) -> None:

        if os.path.exists(zarr_file):
            dataset = zarr.open(zarr_file, mode='a')
        else:
            dataset = zarr.open(zarr_file, mode='w')
            dataset.attrs['name'] = self.name
            dataset.attrs['context'] = str(self.context)
            dataset.attrs['source'] = str(self.source)

        epochs = []
        values = []
        timestamps = []

        for epoch, items in self.epochDataList.items():
            for value, timestamp in items:
                epochs.append(epoch)
                values.append(value)
                timestamps.append(timestamp)

        if 'epochs' not in dataset:
            dataset.create_array('epochs', shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype='i4', compressors=compressor_to_type(self.use_compressor))
            dataset.create_array('values', shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype='f4', compressors=compressor_to_type(self.use_compressor))
            dataset.create_array('timestamps', shape=(0,), chunks=(ZARR_CHUNK_SIZE,), dtype='i8', compressors=compressor_to_type(self.use_compressor))

        dataset['epochs'].append(epochs)
        dataset['values'].append(values)
        dataset['timestamps'].append(timestamps)

        dataset.store.close()

    def save_to_txt(self, txt_file: str,  csv_separator : str = ",") -> None:
        file_exists = os.path.exists(txt_file)
    
        with open(txt_file, "a") as f:
            if not file_exists:
                f.write(f"{self.name}{csv_separator}{self.context}{csv_separator}{self.source}\n")
            
            lines = []
            for epoch, values in self.epochDataList.items():
                for value, timestamp in values:
                    lines.append(f"{epoch}{csv_separator}{value}{csv_separator}{timestamp}")
            f.write("\n".join(lines) + "\n")
