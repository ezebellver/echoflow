import pandas as pd
import matplotlib.pyplot as plt


# Function to plot Precision-Recall curve using precision and recall per epoch
def plot_precision_recall_curve(df_yolov5, df_yolov10):
    # Extract precision and recall values for each epoch for both models
    yolov5_precision = df_yolov5['   metrics/precision']  # Adjust column name as needed
    yolov5_recall = df_yolov5['      metrics/recall']  # Adjust column name as needed
    yolov10_precision = df_yolov10['   metrics/precision(B)']  # Adjust column name as needed
    yolov10_recall = df_yolov10['      metrics/recall(B)']  # Adjust column name as needed

    # Plotting Precision vs Recall for both YOLOv5 and YOLOv10
    plt.figure(figsize=(10, 6))

    # YOLOv5 Precision-Recall plot
    plt.plot(yolov5_recall, yolov5_precision, label='YOLOv5 PR Curve', color='blue', marker='o', linestyle='-',
             markersize=5)

    # YOLOv10 Precision-Recall plot
    plt.plot(yolov10_recall, yolov10_precision, label='YOLOv10 PR Curve', color='green', marker='x', linestyle='--',
             markersize=5)

    # Labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison: YOLOv5 vs. YOLOv10')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both', axis='both', linestyle='-', linewidth=0.1)

    # Show the plot
    plt.show()


# Read YOLOv5 and YOLOv10 results from CSV files
df_yolov5 = pd.read_csv('../../../yolov5/runs/train/my_yolov5_model5/results.csv')
df_yolov10 = pd.read_csv('../../../runs/detect/train4/results.csv')

# Call function to plot the precision-recall curves
plot_precision_recall_curve(df_yolov5, df_yolov10)
