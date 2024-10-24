
import os
import json
import argparse
import prov.model as prov

from prov4ml.utils.file_utils import save_prov_file
    

class Summarizer(): 
    def __init__(self) -> None:
        self.data = {}

    def add_metric_data(self, metric_name, metric_value): 
        self.data[metric_name] = metric_value

    def get_metrics(self): 
        return self.data.keys()

    def get_summary_entity(self, doc): 
        metrics_stats = {}
        for metric in self.get_metrics(): 
            m_value = sum(self.data[metric]) / len(self.data[metric])
            std_value = sum([(x - m_value) ** 2 for x in self.data[metric]]) / len(self.data[metric])

            metrics_stats[f"{metric}_mean"] = m_value,
            metrics_stats[f"{metric}_std"] = std_value

        doc.entity('Metric Summary', other_attributes=metrics_stats)

def main(args):
    experiment_path = args.experiment_path

    experiment_dir = os.path.dirname(experiment_path)
    prov_files = [f for f in os.listdir(experiment_path) if f.endswith(".json")]

    doc = prov.ProvDocument()
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('prov-ml', 'prov-ml')

    summarizer = Summarizer() if args.create_summary else None

    # join the provenance data of all runs
    nsp = None
    for f in prov_files:
        f = os.path.join(experiment_path, f)
        data = json.load(open(f, 'r'))
        # prov_doc = prov.ProvDocument()
        # prov_doc = prov_doc.deserialize(f)

        # get the custom namespace of the experiment
        nsp = data["prefixes"]["namespace"]

        gr = f.split("_")[-1].split(".")[0]

        if summarizer is not None:
            metrics = [m for m in data["entity"].keys() if "TRAINING" in m]
            for metric in metrics:
                values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
                summarizer.add_metric_data(metric, sum(values) / len(values))

        doc.entity(f'{experiment_path}', other_attributes={
            "prov-ml:type": "ProvMLFile",
            "prov-ml:label": f,
            "prov-ml:global_rank": gr, 
        })

    doc.set_default_namespace(nsp)
    if summarizer is not None:
        summarizer.get_summary_entity(doc)

    save_prov_file(
        doc, 
        os.path.join(experiment_dir, "prov_collection.json"), 
        args.create_dot, 
        args.create_svg
    )

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create summary collection of an experiment')
    parser.add_argument('experiment_path', type=str, help='The path to the experiment directory')
    
    parser.add_argument('create_summary', action='store_true', help='Whether to create a metric summary', default=False)

    parser.add_argument('create_dot', action='store_true', help='Whether to create a DOT file for visualization', default=False)
    parser.add_argument('create_svg', action='store_true', help='Whether to create an SVG file for visualization', default=False)
    args = parser.parse_args()

    main(args)