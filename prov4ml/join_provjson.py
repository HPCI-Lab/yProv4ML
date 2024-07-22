
from prov.model import ProvDocument

def join_provjson(runs : list, path : str = ".") -> None:
    """
    Join the provenance data of several runs.

    Parameters:
    -----------
    runs : list
        List of the runs to join.
    path : str, optional
        Path to the directory containing the provenance data. Default is ".".

    Returns:
    --------
    None
    """
    # Load the provenance data of the first run
    prov1 = ProvDocument()
    prov1 = prov1.deserialize(path + "/" + runs[0] + ".json")

    for run in runs[1:]:
        # Load the provenance data of the second run
        prov2 = ProvDocument()
        prov2 = prov2.deserialize(path + "/" + run + ".json")

        # Merge the two provenance data
        prov1.update(prov2)
    
    # Serialize the merged provenance data
    prov1.serialize("merged.json")
    