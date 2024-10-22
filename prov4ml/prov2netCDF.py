import json
import netCDF4 as nc
import os
import gzip
import shutil

from prov4ml.utils.prov_getters import get_metric, get_metrics

def print_file_size(file_path):
    """Prints the size of the file at the given path in bytes."""
    try:
        # Get the size of the file
        file_size = os.path.getsize(file_path) // 10000 / 100
        print(f"The size of the file '{file_path}' is {file_size} Mb.")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def json_to_netcdf(json_file, netcdf_file):
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
        metrics = get_metrics(data, "TRAINING")
        metrics = [get_metric(data, m) for m in metrics]
    
    # Determine dimensions
    unique_sizes = list(set([len(m) for m in metrics]))
    num_metrics_per_size = {}    
    for metric in metrics: 
        size = len(metric)
        if size in num_metrics_per_size: 
            num_metrics_per_size[size] += 1
        else: 
            num_metrics_per_size[size] = 1
        
    # Create NetCDF file
    dataset = nc.Dataset(netcdf_file, 'w', format='NETCDF4')
 
    groups = {size: dataset.createGroup(f"metric_granularity_{size}") for size in unique_sizes}

    for size, group in groups.items(): 
        group.createDimension('metrics', num_metrics_per_size[size])
        group.createDimension('items', size)

        groups[size] = [
            group, 
            group.createVariable('values', 'f4', ('metrics', 'items')), 
            group.createVariable('timestamps', 'i8', ('metrics', 'items')),
            group.createVariable('epochs', 'i4', ('metrics', 'items')),
        ]

    # Add metadata
    dataset.description = 'Metrics with values, timestamps, and epochs'
    dataset.source = 'Converted from JSON'
    # dataset.processing_date = '2024-09-13'

    # Populate variables with data
    for j, metric in enumerate(metrics):
        len_met = len(metric)

        for i, item in metric.iterrows():
            item = item.to_dict()
            groups[len_met][1][0, i] = item['value']
            groups[len_met][2][0, i] = item['time']
            groups[len_met][3][0, i] = item['epoch']

    # Close the dataset
    dataset.close()
    print(f'NetCDF file "{netcdf_file}" created successfully.')

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
if __name__ == "__main__": 
    json_to_netcdf('test.json', 'test.nc')

    print_file_size('test.json')
    print_file_size('test.nc')    

    # Example usage
    compress_file('test.nc', 'test.nc.gz')
    compress_file('test.json', 'test.json.gz')

    print_file_size('test.json.gz')
    print_file_size('test.nc.gz')