
import os
import prov.model as prov

from ..constants import PROV4ML_DATA
from ..utils.file_utils import save_prov_file
from ..utils.funcs import get_global_rank
    
def create_prov_collection(create_dot=False, create_svg=False): 
    # get all prov files
    prov_files = [f for f in os.listdir(PROV4ML_DATA.EXPERIMENT_DIR) if f.endswith(".json")]

    doc = prov.ProvDocument()
    #set namespaces
    doc.set_default_namespace(PROV4ML_DATA.USER_NAMESPACE)
    doc.add_namespace('prov','http://www.w3.org/ns/prov#')
    doc.add_namespace('xsd','http://www.w3.org/2000/10/XMLSchema#')
    doc.add_namespace('prov-ml', 'prov-ml')

    # join the provenance data of all runs
    for f in prov_files:
        f = PROV4ML_DATA.EXPERIMENT_DIR + "/" + f
        prov_doc = prov.ProvDocument()
        prov_doc = prov_doc.deserialize(f)

        if get_global_rank() is not None:
            gr = f.split("_")[-1].split(".")[0]
        else:
            gr = None 

        doc.entity(f'{PROV4ML_DATA.EXPERIMENT_NAME}', other_attributes={
            "prov-ml:type": "ProvMLFile",
            "prov-ml:label": f,
            "prov-ml:global_rank": gr
        })

    save_prov_file(doc, PROV4ML_DATA.EXPERIMENT_DIR + "/prov_collection.json", create_dot, create_svg)
