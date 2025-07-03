import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

provless = "./provless.txt"
provwith = "./provwith.txt"
provwithout = "./provwithoutmetric.txt"

with open(provless, 'r') as file:
    provless = [float(line.strip()) for line in file if line.strip()]
    provless = pd.DataFrame(provless)

with open(provwith, 'r') as file:
    provwith = [float(line.strip()) for line in file if line.strip()]
    provwith = pd.DataFrame(provwith)

with open(provwithout, 'r') as file:
    provwithout = [float(line.strip()) for line in file if line.strip()]
    provwithout = pd.DataFrame(provwithout)

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
plt.plot(provless, label="No Prov Collection", linewidth=3)
plt.plot(provwithout, label="With Default Provenance", linewidth=3)
plt.plot(provwith, label="Provenance with System Metrics", linewidth=3)
plt.ylabel('Memory Usage (%)')
plt.xlabel("Timesteps")
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show()
