import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
yolov5_simple = pd.read_csv('../model_sv5_metrics.csv')
yolov5_extensive = pd.read_csv('../model_ev5_metrics.csv')
yolov10 = pd.read_csv('../yolov10_metrics.csv')

# Manually define model labels
model_labels = ["YOLOv5 Extensive", "YOLOv5 Simple", "YOLOv10"]

# Define function for calculating average metrics
def calculate_averages(df):
    return {
        "Avg Preprocess Time (ms)": df["preprocess_time"].mean() * 1000,
        "Avg Inference Time (ms)": df["inference_time"].mean() * 1000,
        "Avg FPS": df["fps"].mean(),
        "Avg CPU Usage (%)": df["cpu_usage"].mean(),
        "Avg GPU Usage (%)": df["gpu_usage"].mean()
    }

# Calculate averages for each model
averages = [
    calculate_averages(yolov5_extensive),
    calculate_averages(yolov5_simple),
    calculate_averages(yolov10)
]

# Add model labels to the data
for i, label in enumerate(model_labels):
    averages[i]["Model"] = label

# Convert to DataFrame for easier plotting
averages_df = pd.DataFrame(averages)

# Convert the 'Model' column to an array for plotting purposes
models = averages_df["Model"].values

# Print the averages DataFrame for debugging
print(averages_df)


DayOfWeekOfCall = [1,2,3]
DispatchesOnThisWeekday = [77, 32, 42]

# Plot 1: Preprocessing and Inference Time
plt.figure(figsize=(12, 6))
bar_width = 0.35
x = range(len(models))

# Adjust the positions of the bars for correct alignment
plt.bar(x, averages_df["Avg Preprocess Time (ms)"], bar_width, label="Preprocess Time (ms)")
plt.bar([i + bar_width for i in x], averages_df["Avg Inference Time (ms)"], bar_width, label="Inference Time (ms)")

plt.xlabel("Model")
plt.ylabel("Time (ms)")
plt.title("Preprocessing vs Inference Time Comparison")

print(model_labels)
# Set the x-axis tick positions to the center of the bars and use model labels
#plt.xticks([i + bar_width / 2 for i in x], model_labels)
plt.xticks([0, 1, 2], ["a", "b", "c"])

plt.legend()
plt.tight_layout()
plt.show()

# Plot 2: FPS
plt.figure(figsize=(8, 5))
plt.bar(models, averages_df["Avg FPS"], color='orange')
plt.xlabel("Model")
plt.ylabel("Frames Per Second (FPS)")
plt.title("FPS Comparison")
plt.tight_layout()
plt.show()

# Plot 3: CPU and GPU Usage
plt.figure(figsize=(12, 6))
bar_width = 0.35

# Adjust the positions of the bars for correct alignment
plt.bar(x, averages_df["Avg CPU Usage (%)"], bar_width, label="CPU Usage (%)", color='blue')
plt.bar([i + bar_width for i in x], averages_df["Avg GPU Usage (%)"], bar_width, label="GPU Usage (%)", color='green')

plt.xlabel("Model")
plt.ylabel("Usage (%)")
plt.title("CPU vs GPU Usage Comparison")

# Set the x-axis tick positions to the center of the bars and use model labels
plt.xticks([i + bar_width / 2 for i in x], models)

plt.legend()
plt.tight_layout()
plt.show()
