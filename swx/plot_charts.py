
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

file_path = "./swx/plot_data.json"

data = json.load(open(file_path))

def timestamp_to_seconds(ts):
    return ts / 1000

def get_metric(data, metric, time_in_sec=False, time_incremental=False):
    try: 
        epochs = eval(data["entity"][metric]["prov-ml:metric_epoch_list"])
        values = eval(data["entity"][metric]["prov-ml:metric_value_list"])
        times = eval(data["entity"][metric]["prov-ml:metric_timestamp_list"])
    except: 
        return pd.DataFrame(columns=["epoch", "value", "time"])
    
    # convert to minutes and sort
    if time_in_sec:
        times = [timestamp_to_seconds(ts) for ts in times]
        
    df = pd.DataFrame({"epoch": epochs, "value": values, "time": times}).drop_duplicates()

    if time_incremental: 
        df["time"] = df["time"].diff().fillna(0)

    df = df.sort_values(by="time")
    return df

def get_metrics(data, keyword=None):
    ms = data["entity"].keys()
    if keyword is None:
        return ms
    else:
        return [m for m in ms if keyword in m]

print(get_metrics(data, keyword="TRAINING"))

sns.set(style="darkgrid", palette="colorblind", font_scale=1.2)

# Get metrics
loss1 = get_metric(data, "Loss_Context.TRAINING")["value"]
cpu_e1 = get_metric(data, "cpu_usage_Context.TRAINING")["value"]

loss2 = get_metric(data, "cpu_power_Context.TRAINING")["value"]
cpu_e2 = get_metric(data, "energy_consumed_Context.TRAINING")["value"]

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Training Loss vs CPU Usage
ax1 = axes[0]
color1, color2 = sns.color_palette("colorblind", 2)

ax1.plot(loss1, color=color1, label="Training Loss", linewidth=2)
ax1.set_ylabel("Training Loss", color=color1)
ax1.set_title("Training Loss vs CPU Usage")
ax1.set_xlabel("Epoch")
ax1.tick_params(axis='y', labelcolor=color1)

ax1b = ax1.twinx()
ax1b.plot(cpu_e1, color=color2, label="CPU Usage (%)", linewidth=2)
ax1b.set_ylabel("CPU Usage (%)", color=color2)
ax1b.tick_params(axis='y', labelcolor=color2)

# Plot 2: CPU Power vs Energy Consumed
ax2 = axes[1]
color3, color4 = sns.color_palette("colorblind", 2)

ax2.plot(loss2, color=color3, label="CPU Power (W)", linewidth=2)
ax2.set_ylabel("CPU Power (W)", color=color3)
ax2.set_title("CPU Power vs Energy Consumed")
ax2.set_xlabel("Epoch")
ax2.tick_params(axis='y', labelcolor=color3)

ax2b = ax2.twinx()
ax2b.plot(cpu_e2, color=color4, label="Energy Consumed (J)", linewidth=2)
ax2b.set_ylabel("Energy Consumed (J)", color=color4)
ax2b.tick_params(axis='y', labelcolor=color4)

plt.tight_layout()
plt.show()