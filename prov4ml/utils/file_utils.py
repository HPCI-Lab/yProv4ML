
import os
import prov.dot as dot
import prov.model as prov

from ..constants import PROV4ML_DATA

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

    if create_graph:
        dot_filename = os.path.basename(prov_file).replace(".json", ".dot")
        path_dot = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, dot_filename)
        with open(path_dot, 'w') as prov_dot:
            prov_dot.write(dot.prov_to_dot(doc).to_string())

    if create_svg:
        svg_filename = os.path.basename(prov_file).replace(".json", ".svg")
        path_svg = os.path.join(PROV4ML_DATA.EXPERIMENT_DIR, svg_filename)
        os.system(f"dot -Tsvg {path_dot} > {path_svg}")