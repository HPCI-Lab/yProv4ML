import matplotlib.pyplot as plt
import sys
sys.path.append("../yProv4ML")

import yprov4ml

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="plot_example", 
    provenance_save_dir="prov",
    collect_all_processes=False, 
    disable_codecarbon=True, 
)

yprov4ml.log_source_code()

data = [1, 2, 3, 4, 5]

plt.plot(data)
plt.savefig("tmp.png")

yprov4ml.log_artifact("tmp", "tmp.png")

yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False)