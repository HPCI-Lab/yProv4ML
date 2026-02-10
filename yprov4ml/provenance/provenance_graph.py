
import os
import prov
import prov.model as prov
from rocrate.rocrate import ROCrate
from pathlib import Path

from yprov4ml.constants import PROV4ML_DATA

import os
import prov.model as prov

from yprov4ml.utils.prov_utils import custom_prov_to_dot

def create_prov_document() -> prov.ProvDocument:
    
    doc = PROV4ML_DATA.root_provenance_doc

    for (name, ctx) in PROV4ML_DATA.metrics.keys():
        source = PROV4ML_DATA.metrics[(name, ctx)].source
        metric_file_path = os.path.join(PROV4ML_DATA.METRIC_DIR, f"{name}_{str(ctx)}_{str(source)}_GR{PROV4ML_DATA.global_rank}.{PROV4ML_DATA.metrics_file_type}")
        e = PROV4ML_DATA.add_artifact(name,metric_file_path,0,ctx, source, is_input=False, log_copy_in_prov_directory=False)
        
        e.add_attributes({
            f'{PROV4ML_DATA.LABEL_PREFIX}:context': str(ctx),
            f'{PROV4ML_DATA.LABEL_PREFIX}:source': str(source)
        })

    return doc


def save_prov_file(
        doc : prov.ProvDocument,
        prov_file : str,
        create_graph : bool =False, 
        create_svg : bool =False
    ) -> None:
    """
    Save the provenance document to a file.

    Parameters:
    -----------
    doc : prov.ProvDocument
        The provenance document to save.
    prov_file : str
        The path to the file where the provenance document will be saved.
    create_graph : bool 
        A flag to indicate if a graph should be created. Defaults to False.
    create_svg : bool
        A flag to indicate if an SVG should be created. Defaults to False.
    
    Returns:
        None
    """

    with open(prov_file, 'w') as prov_graph:
        doc.serialize(prov_graph)

    if create_svg and not create_graph:
        raise ValueError("Cannot create SVG without creating the graph.")

    if create_graph:
        dot_filename = os.path.basename(prov_file).replace(".json", ".dot")
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(custom_prov_to_dot(doc).to_string())

    if create_svg:
        svg_filename = os.path.basename(prov_file).replace(".json", ".svg")
        path_svg = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, svg_filename)
        os.system(f"dot -Tsvg {path_dot} > {path_svg}")



def get_properties_from_file(file : str):
    if file.endswith(".dot"): 
        return {
            "name": "pygraphviz provenance graph file",
            "encodingFormat": "application/dot"
        }
    elif file.endswith(".csv"): 
        return {
            "name": "metric",
            "encodingFormat": "text/csv"
        }
    elif file.endswith(".svg"): 
        return {
            "name": "pygraphviz svg provenance graph file",
            "encodingFormat": "image/svg+xml"
        }
    elif file.endswith(".json") and "/" not in file: 
        return {
            "name": "provenance JSON file",
            "encodingFormat": "text/json"
        }
    elif file.endswith(".json") and "/" in file: 
        return {
            "name": "JSON property description",
            "encodingFormat": "text/json"
        }
    elif file.endswith(".pt") or file.endswith(".pth"): 
        return {
            "name": "pytorch model checkpoint",
            "encodingFormat": "application/octet-stream"
        }
    elif file.endswith(".py"): 
        return {
            "name": "python source file",
            "encodingFormat": "text/plain"
        }
    else: 
        return {
            "name": file,
            "encodingFormat": f"{file.split('.')[-1]}",
        }

def create_rocrate_in_dir(directory): 
    crate = ROCrate()

    for (d, _, fs) in os.walk(directory): 
        for f in fs: 
            file_path = d + "/" + f
            if Path(file_path).exists():
                property = get_properties_from_file(file_path)
                property["@type"] = "File" 
                property["@id"] = file_path
                crate.add_file(file_path, dest_path=file_path, properties=property)

    # crate.write("exp_crate")
    crate.write_zip(f"{directory}.zip")