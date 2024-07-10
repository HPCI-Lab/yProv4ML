
import os
import json
from prov.model import ProvDocument

from ..constants import PROV4ML_DATA
    
def create_prov_collection(): 
    # get all prov files
    prov_files = [f for f in os.listdir(PROV4ML_DATA.EXPERIMENT_DIR) if f.endswith(".json")]

    metadata_prov = {
        "prefix": {
            "xsd_1": "http://www.w3.org/2000/10/XMLSchema#",
            "prov-ml": "prov-ml",
            "default": PROV4ML_DATA.USER_NAMESPACE,
        },
        "entity": [],
        "activity": {},
        "wasGeneratedBy": {},
        "agent": {},
        "wasAssociatedWith": {}
    }

    # join the provenance data of all runs
    for f in prov_files:
        f = PROV4ML_DATA.EXPERIMENT_DIR + "/" + f
        prov = ProvDocument()
        prov = prov.deserialize(f)

        if os.environ.get("SLURM_JOB_ID"):
            gr = f.split("_")[-1].split(".")[0]
        else:
            gr = None

        file_entity = {
            "prov-ml:type": "ProvMLFile",
            "prov-ml:label": f,
            "prov-ml:global_rank": gr
        }

        metadata_prov["entity"].append(file_entity)

    # serialize the provenance collection (metadata_prov)
    json.dump(metadata_prov, open(PROV4ML_DATA.EXPERIMENT_DIR + "/prov_collection.json", "w"), indent=2)
    

