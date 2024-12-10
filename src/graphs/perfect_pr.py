import matplotlib.pyplot as plt
import numpy as np

# Define the points for a perfect PR curve
recall = np.array([0, 1, 1])  # Recall values: starts at 0, then 1, then 1
precision = np.array([1, 1, 0])  # Precision: stays at 1, then drops to 0

# Plotting the Precision-Recall curve
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, label='Perfect PR Curve', color='blue', linestyle='-', marker='o')

# Labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Perfect Precision-Recall Curve')
plt.legend()

# Barely visible grid
plt.grid(True, alpha=0.5, which='both', axis='both', linestyle='-', linewidth=0.1)

# Show the plot
plt.show()
