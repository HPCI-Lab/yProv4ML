import json
import netCDF4 as nc
import os

from utils.prov_getters import get_metric, get_metrics

def print_file_size(file_path):
    """Prints the size of the file at the given path in bytes."""
    try:
        # Get the size of the file
        file_size = os.path.getsize(file_path)
        print(f"The size of the file '{file_path}' is {file_size} bytes.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def json_to_netcdf(json_file, netcdf_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

        metrics = ['gpu_memory_usage_Context.TRAINING', 'emissions_rate_Context.TRAINING', 'gpu_energy_Context.TRAINING', 'custom_emission_perc_diff_Context.TRAINING', 'ram_energy_Context.TRAINING', 'energy_consumed_Context.TRAINING', 'emissions_Context.TRAINING', 'disk_usage_Context.TRAINING', 'MSE_test_Context.EVALUATION', 'cpu_power_Context.TRAINING', 'gpu_power_Context.TRAINING', 'custom_emissions_Context.TRAINING', 'MSE_train_Context.TRAINING', 'ram_power_Context.TRAINING', 'gpu_temperature_Context.TRAINING', 'memory_usage_Context.TRAINING', 'gpu_usage_Context.TRAINING', 'cpu_energy_Context.TRAINING', 'cpu_usage_Context.TRAINING', 'gpu_power_usage_Context.TRAINING']

        metrics = [get_metric(data, m) for m in metrics]
        max_len = max([len(m) for m in metrics])
    
    # Determine dimensions
    num_metrics = len(metrics)
    num_items = max_len if num_metrics > 0 else 0

    # Create NetCDF file
    dataset = nc.Dataset(netcdf_file, 'w', format='NETCDF4')

    # Create dimensions
    dataset.createDimension('metrics', num_metrics)
    dataset.createDimension('items', num_items)

    # Create variables
    values = dataset.createVariable('values', 'f4', ('metrics', 'items'))
    timestamps = dataset.createVariable('timestamps', 'i8', ('metrics', 'items'))
    epochs = dataset.createVariable('epochs', 'i4', ('metrics', 'items'))

    # Add metadata
    dataset.description = 'Metrics with values, timestamps, and epochs'
    dataset.source = 'Converted from JSON'
    dataset.processing_date = '2024-09-13'

    # Populate variables with data
    for j, metric in enumerate(metrics):
        for i, item in metric.iterrows():
            item = item.to_dict()
            values[j, i] = item['value']
            timestamps[j, i] = item['time']
            epochs[j, i] = item['epoch']

    # Close the dataset
    dataset.close()
    print(f'NetCDF file "{netcdf_file}" created successfully.')

# Example usage
json_to_netcdf('test.json', 'test.nc')

print_file_size('test.json')
print_file_size('test.nc')

import gzip
import shutil

def compress_file(input_file, output_file):
    """Compress a file using gzip."""
    try:
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"File '{input_file}' compressed to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
compress_file('test.nc', 'test.nc.gz')
compress_file('test.json', 'test.json.gz')

print_file_size('test.json.gz')
print_file_size('test.nc.gz')