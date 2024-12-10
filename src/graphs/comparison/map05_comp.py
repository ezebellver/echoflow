import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate F1 score
def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

# Read YOLOv5 and YOLOv10 results from CSV files
# Replace 'yolov5_results.csv' and 'yolov10_results.csv' with your actual CSV file paths
df_yolov5 = pd.read_csv('../../../yolov5/runs/train/my_yolov5_model5/results.csv')
df_yolov10 = pd.read_csv('../../../runs/detect/train4/results.csv')

# Plotting mAP@0.5
plt.figure(figsize=(12, 6))
plt.plot(df_yolov5['               epoch'], df_yolov5['     metrics/mAP_0.5'], label='YOLOv5 mAP@0.5', color='blue', marker='o')
plt.plot(df_yolov10['                  epoch'], df_yolov10['       metrics/mAP50(B)'], label='YOLOv10 mAP@0.5', color='green', marker='x')

plt.xlabel('Epoch')
plt.ylabel('mAP@0.5')
plt.title('mAP@0.5 Comparison: YOLOv5 vs. YOLOv10')
plt.legend(loc='lower right')
# Set y axis limit from 0 to 1
plt.ylim(0, 1)
plt.xlim(0, 51)

plt.grid(True, alpha=0.5, which='both', axis='both', linestyle='-', linewidth=0.1)
plt.show()