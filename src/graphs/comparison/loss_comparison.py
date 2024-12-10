import pandas as pd
import matplotlib.pyplot as plt

# Load YOLOv5 and YOLOv10 results from CSV files
# Replace these paths with your actual CSV file paths
df_yolov5 = pd.read_csv('../../../yolov5/runs/train/my_yolov5_model5/results.csv')
df_yolov10 = pd.read_csv('../../../runs/detect/train4/results.csv')

# Plotting Box Loss
plt.figure(figsize=(12, 6))
plt.plot(df_yolov5['               epoch'], df_yolov5['      train/box_loss'], label='YOLOv5 Train Box Loss', color='blue', marker='o')
plt.plot(df_yolov10['                  epoch'], df_yolov10['         train/box_loss'], label='YOLOv10 Train Box Loss', color='green', marker='x')
plt.plot(df_yolov5['               epoch'], df_yolov5['        val/box_loss'], label='YOLOv5 Val Box Loss', color='blue', linestyle='--')
plt.plot(df_yolov10['                  epoch'], df_yolov10['           val/box_loss'], label='YOLOv10 Val Box Loss', color='green', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Box Loss')
plt.title('Box Loss Comparison: YOLOv5 vs. YOLOv10')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.5, linestyle='--')
plt.show()


# Plotting Classification Loss
plt.figure(figsize=(12, 6))
plt.plot(df_yolov5['               epoch'], df_yolov5['      train/cls_loss'], label='YOLOv5 Train Classification Loss', color='blue', marker='o')
plt.plot(df_yolov10['                  epoch'], df_yolov10['         train/cls_loss'], label='YOLOv10 Train Classification Loss', color='green', marker='x')
plt.plot(df_yolov5['               epoch'], df_yolov5['        val/cls_loss'], label='YOLOv5 Val Classification Loss', color='blue', linestyle='--')
plt.plot(df_yolov10['                  epoch'], df_yolov10['           val/cls_loss'], label='YOLOv10 Val Classification Loss', color='green', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Classification Loss')
plt.title('Classification Loss Comparison: YOLOv5 vs. YOLOv10')
plt.legend(loc='upper right')
# plt.ylim(0, 1)
plt.xlim(0, 51)

plt.grid(True, alpha=0.5, which='both', axis='both', linestyle='-', linewidth=0.1)
plt.show()
