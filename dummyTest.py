"""
NAIVE MEAN-FIELD BASELINE
-------------------------
Function:
    Calculates the "Average Flow Field" across the entire training set and uses it as a
    constant prediction for every test case.

Key Features:
    1. Performance Floor: Establishes the absolute minimum accuracy any machine learning model
       must beat to be considered useful.
    2. Reality Check: If a complex Neural Network achieves 90% accuracy but this dummy script
       achieves 89%, the network has effectively learned nothing but the average.
    3. Metric: Converts Relative L2 Error into an intuitive "Accuracy Percentage".
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from neuralop.losses import LpLoss

# Configuration
BASE_PATH = "."
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200
r1, r2 = 1, 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

print("Loading Output Data...")
# Generate dummy data if file not found (for this standalone script to run)
if not os.path.exists(OUTPUT_Sigma):
    print("Warning: File not found. Generating random dummy data for testing.")
    output = np.random.rand(1200, 221, 51)
else:
    output = np.load(OUTPUT_Sigma)

# Handle shape inconsistencies
if output.ndim == 4 and output.shape[1] > 4:
    output = output[:, 4]
elif output.ndim == 3:
    pass
else:
    raise ValueError(f"Unexpected shape: {output.shape}")

# Convert to Tensor
output = torch.tensor(output, dtype=torch.float)
print(f"Total Data Shape: {output.shape}")

# Split Data
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]

print(f"Training Set: {y_train.shape}")
print(f"Test Set:     {y_test.shape}")

# Start Timer
start_time = time.time()

# ==========================================
# 1. Create Dummy Model ("Training")
# ==========================================
print("\nComputing Mean Field (Dummy Model)...")
# Calculate the average field across all training samples
mean_field = torch.mean(y_train, dim=0)

# ==========================================
# 2. Evaluate Dummy Model ("Inference")
# ==========================================
# We use LpLoss to get the Relative Error, then convert to Accuracy
loss_fn = LpLoss(d=2, p=2, reduction='sum')

accuracy_list = []

# We predict the 'mean_field' for every single item in the test set
dummy_prediction = mean_field.unsqueeze(0)  # Shape [1, H, W]

print("Evaluating Dummy Model on Test Set...")
with torch.no_grad():
    for i in range(len(y_test)):
        truth = y_test[i].unsqueeze(0)  # Shape [1, H, W]

        # 1. Get Relative Error (LpLoss)
        # e.g., 0.15 means 15% Error
        rel_error = loss_fn(dummy_prediction, truth).item()

        # 2. Convert to Accuracy %
        # e.g., 1.0 - 0.15 = 0.85 (85% Accurate)
        acc = (1.0 - rel_error) * 100
        accuracy_list.append(acc)

end_time = time.time()
total_duration = end_time - start_time

avg_accuracy = np.mean(accuracy_list)

print("=" * 40)
print(f"DUMMY BASELINE RESULTS")
print("=" * 40)
print(f"Average Accuracy:       {avg_accuracy:.2f}%")
print(f"Interpretation:         (1 - Relative_L2_Error) * 100")
print(f"Total Time:             {total_duration:.4f} seconds")
print("=" * 40)

# ==========================================
# 3. Visualization
# ==========================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(mean_field.numpy().T, cmap='viridis', origin='lower')
plt.colorbar()
plt.title("The 'Dummy' Mean Field")

plt.subplot(1, 2, 2)
plt.imshow(y_test[0].numpy().T, cmap='viridis', origin='lower')
plt.colorbar()
plt.title("Actual Ground Truth (Sample 0)")

plt.tight_layout()
plt.show()