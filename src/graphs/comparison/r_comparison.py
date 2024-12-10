import pandas as pd
import matplotlib.pyplot as plt

# Read YOLOv5 and YOLOv10 results from CSV files
# Replace 'yolov5_results.csv' and 'yolov10_results.csv' with your actual CSV file paths
df_yolov5 = pd.read_csv('../../../yolov5/runs/train/my_yolov5_model5/results.csv')
df_yolov10 = pd.read_csv('../../../runs/detect/train4/results.csv')

# Plotting F1 scores for both YOLOv5 and YOLOv10
plt.figure(figsize=(10, 6))
plt.plot(df_yolov5['               epoch'], df_yolov5['      metrics/recall'], label='YOLOv5 Recall', color='blue', marker='o')
plt.plot(df_yolov10['                  epoch'], df_yolov10['      metrics/recall(B)'], label='YOLOv10 Recall', color='green', marker='x')

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall Comparison: YOLOv5 vs. YOLOv10')
plt.legend()

# Set y axis limit from 0 to 1
plt.ylim(0, 1)
plt.xlim(0, 51)
# Barely visible grid
plt.grid(True, alpha=0.5, which='both', axis='both', linestyle='-', linewidth=0.1)

# Show the plot
plt.show()
