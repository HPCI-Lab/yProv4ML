
from prov.model import ProvDocument

def join_provjson(runs : list, path : str = "."):
    """
    Join the provenance data of several runs.
    """
    # path = "/Users/gabrielepadovani/Desktop/Università/PhD/provenance/prov"
    # Load the provenance data of the first run
    # r1 = str(runs[0]).split("_")[1].split(".")[0]
    prov1 = ProvDocument()
    prov1 = prov1.deserialize(path + "/" + runs[0] + ".json")

    for run in runs[1:]:
        # Load the provenance data of the second run
        # r2 = str(run).split("_")[1].split(".")[0]
        prov2 = ProvDocument()
        prov2 = prov2.deserialize(path + "/" + run + ".json")

        # Merge the two provenance data
        prov1.update(prov2)
    
    # Serialize the merged provenance data
    prov1.serialize("merged.json")
    