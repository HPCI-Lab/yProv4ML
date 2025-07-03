import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

epochs = [2, 5, 10, 15]
provless = [10.298, 23.7071, 46.6646, 68.6635]
provwith = [16.1814, 29.7835, 52.093, 75.5886]
provwithsystem = [37.242, 81.5524, 155.80788, 243.5721]

df = pd.DataFrame({
    'Epochs': epochs * 3,
    'Time (s)': provless + provwith + provwithsystem,
    'Setup': ['No Prov Collection']*len(epochs) + ['Default Provenance']*len(epochs) + ['Provenance with System Metrics']*len(epochs)
})

sns.set(style="whitegrid", font_scale=1.2)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Epochs', y='Time (s)', hue='Setup', marker='o', linewidth=3)
plt.tight_layout()
plt.legend()
plt.grid(True)
plt.show() 