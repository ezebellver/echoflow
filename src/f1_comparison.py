import pandas as pd
import matplotlib.pyplot as plt

# Function to calculate F1 score
def calculate_f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

# Read YOLOv5 and YOLOv10 results from CSV files
# Replace 'yolov5_results.csv' and 'yolov10_results.csv' with your actual CSV file paths
df_yolov5 = pd.read_csv('../yolov5/runs/train/my_yolov5_model2/results.csv')
df_yolov10 = pd.read_csv('../runs/detect/train4/results.csv')

# Calculate F1 scores for each epoch
df_yolov5['f1'] = calculate_f1(df_yolov5['   metrics/precision'], df_yolov5['      metrics/recall'])
df_yolov10['f1'] = calculate_f1(df_yolov10['   metrics/precision(B)'], df_yolov10['      metrics/recall(B)'])

# Plotting F1 scores for both YOLOv5 and YOLOv10
plt.figure(figsize=(10, 6))
plt.plot(df_yolov5['               epoch'], df_yolov5['f1'], label='YOLOv5 F1', color='blue', marker='o')
plt.plot(df_yolov10['                  epoch'], df_yolov10['f1'], label='YOLOv10 F1', color='green', marker='x')

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score Comparison: YOLOv5 vs. YOLOv10')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
