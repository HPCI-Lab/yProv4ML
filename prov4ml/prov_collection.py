
import os
import argparse
import prov.model as prov

from .utils.file_utils import save_prov_file
    

def main(experiment_path : str, create_dot : bool = False, create_svg : bool = False): 
    experiment_dir = os.path.dirname(experiment_path)
    prov_files = [f for f in os.listdir(experiment_path) if f.endswith(".json")]

    doc = prov.ProvDocument()
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('prov-ml', 'prov-ml')

    # join the provenance data of all runs
    nsp = None
    for f in prov_files:
        f = os.path.join(experiment_path, f)
        prov_doc = prov.ProvDocument()
        prov_doc = prov_doc.deserialize(f)

        # get the custom namespace of the experiment
        nsp = prov_doc.namespaces[0]

        gr = f.split("_")[-1].split(".")[0]

        doc.entity(f'{experiment_path}', other_attributes={
            "prov-ml:type": "ProvMLFile",
            "prov-ml:label": f,
            "prov-ml:global_rank": gr
        })

    doc.set_default_namespace(nsp)

    save_prov_file(
        doc, 
        os.path.join(experiment_dir, "prov_collection.json"), 
        create_dot, 
        create_svg
    )

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create summary collection of an experiment')
    parser.add_argument('experiment_path', type=str, help='The path to the experiment directory')
    parser.add_argument('--create_dot', action='store_true', help='Whether to create a DOT file for visualization', default=False)
    parser.add_argument('--create_svg', action='store_true', help='Whether to create an SVG file for visualization', default=False)
    args = parser.parse_args()

    main(args.experiment_path, args.create_dot, args.create_svg)