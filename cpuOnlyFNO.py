"""
CPU-ONLY FNO REFERENCE IMPLEMENTATION
-------------------------------------
Function:
    A streamlined implementation of the Fourier Neural Operator (FNO) specifically configured
    for CPU execution, bypassing CUDA-specific optimizations. This is for running on computers
    without a GPU, or with a GPU that doesn't have a lot of VRAM (Like mine)

Key Features:
    1. Stability: Disables Automatic Mixed Precision (AMP) and gradient scaling to prevent CPU crashes.
    2. Efficiency: Retains the "Padded Dataset" strategy (221x51 -> 224x64) to ensure FFT operations
       run on efficient, even-sized grids.
    3. Baseline: Serves as the stable "Gold Standard" implementation for verifying model correctness
       when GPU resources are unavailable or debugging is required.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gc
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

# --- neuraloperator imports ---
from neuralop.models import FNO
from neuralop.losses import LpLoss

# ---------------------------------------
# Configuration
# ---------------------------------------
torch.manual_seed(0)
np.random.seed(0)

# Force CPU
device = torch.device("cpu")
print(f" forcing device to: {device}")

BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")

FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "fno_cpu_model.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, "fno_training_log_cpu.xlsx")
FINAL_PLOT_FILENAME = "fno_final_comparison_cpu.png"

os.makedirs(MODEL_PATH, exist_ok=True)

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200

# Batch size can be slightly higher on CPU usually,
# but keeping it low ensures stability.
batch_size = 32
learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5

modes = 12
width = 32

################################################################
# Data Loading & Preprocessing
################################################################
print("Loading data...")
inputX = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
# Shape: [Samples, 221, 51, 2]
input_data = torch.stack([inputX, inputY], dim=-1)

output_data = np.load(OUTPUT_Sigma)
if output_data.ndim == 4 and output_data.shape[1] > 4:
    output_data = output_data[:, 4]
elif output_data.ndim == 3:
    pass
else:
    raise ValueError(f"Unexpected shape for OUTPUT_Sigma: {output_data.shape}")
output_data = torch.tensor(output_data, dtype=torch.float)

# Split into train/test
x_train = input_data[:ntrain]
x_test = input_data[ntrain:ntrain + ntest]
y_train = output_data[:ntrain].unsqueeze(-1)
y_test = output_data[ntrain:ntrain + ntest].unsqueeze(-1)

# Permute to [Batch, Channels, Height, Width] for FNO
x_train = x_train.permute(0, 3, 1, 2).contiguous()
x_test = x_test.permute(0, 3, 1, 2).contiguous()
y_train = y_train.permute(0, 3, 1, 2).contiguous()
y_test = y_test.permute(0, 3, 1, 2).contiguous()

print(f"Train Input Shape: {x_train.shape}")


# --- PADDED DATASET (Efficient FFT) ---
class PaddedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Target even dimensions
        self.target_h = 224
        self.target_w = 64

        # Calculate padding needed
        self.pad_h = self.target_h - x.shape[2]
        self.pad_w = self.target_w - x.shape[3]

        # Create grid for the TARGET size
        gridx = torch.tensor(np.linspace(0, 1, self.target_h), dtype=torch.float)
        gridx = gridx.reshape(1, self.target_h, 1).repeat([1, 1, self.target_w])
        gridy = torch.tensor(np.linspace(0, 1, self.target_w), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.target_w).repeat([1, self.target_h, 1])
        self.grid = torch.cat((gridx, gridy), dim=0)  # [2, 224, 64]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_raw = self.x[idx]
        y_raw = self.y[idx]

        # Pad to even size
        x_padded = F.pad(x_raw, (0, self.pad_w, 0, self.pad_h))
        y_padded = F.pad(y_raw, (0, self.pad_w, 0, self.pad_h))

        # Concat Grid
        x_combined = torch.cat((x_padded, self.grid), dim=0)

        return {"x": x_combined, "y": y_padded}


train_loader = DataLoader(PaddedDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(PaddedDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(PaddedDataset(x_test, y_test), batch_size=1, shuffle=False)

################################################################
# Model Definition
################################################################
# Note: domain_padding is None because we manually padded in the dataset
model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=None,
    non_linearity=torch.nn.GELU()
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(d=2, p=2, reduction='sum')

################################################################
# Training Loop (Standard CPU)
################################################################
training_history = []

print("Starting training on CPU...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0

    current_lr = optimizer.param_groups[0]['lr']

    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        # Standard forward/backward (No AMP on CPU)
        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            out = model(x)
            test_l2 += loss_fn(out, y).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    epoch_time = t2 - t1

    print(f"Epoch {ep + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_l2:.6f} | Test Loss: {test_l2:.6f}")

    epoch_data = {
        'epoch': ep + 1,
        'time_s': epoch_time,
        'train_loss': train_l2,
        'test_loss': test_l2,
        'learning_rate': current_lr
    }
    training_history.append(epoch_data)

################################################################
# Save
################################################################
try:
    pd.DataFrame(training_history).to_excel(EXCEL_FILENAME, index=False)
    print("History saved.")
except Exception as e:
    print(f"Error saving history: {e}")

torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Saved model to {FINAL_MODEL_FILENAME}")

################################################################
# Plotting
################################################################
print("Generating comparison plot...")

final_model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=None,
    non_linearity=torch.nn.GELU()
).to(device)

final_model.load_state_dict(torch.load(FINAL_MODEL_FILENAME, map_location=device,weights_only=False))
final_model.eval()

with torch.no_grad():
    batch = next(iter(test_loader2))
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    out = final_model(x)

    # Unpad for visualization: Slice back to [221, 51]
    valid_h = 221
    valid_w = 51

    X = x[0, 0, :valid_h, :valid_w].numpy()
    Y = x[0, 1, :valid_h, :valid_w].numpy()
    truth = y[0, 0, :valid_h, :valid_w].numpy()
    pred = out[0, 0, :valid_h, :valid_w].numpy()

    # Plotting
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    # Full Views
    im = ax[0, 0].pcolormesh(X, Y, truth, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title('Ground Truth (Full)')

    im = ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title('Prediction (Full)')

    im = ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title('Difference (Full)')

    # Zoom logic (approximate center)
    nx, ny = 20, 10
    X_small = X[nx:-nx, :ny]
    Y_small = Y[nx:-nx, :ny]
    truth_small = truth[nx:-nx, :ny]
    pred_small = pred[nx:-nx, :ny]

    vmin_small, vmax_small = np.min(truth_small), np.max(truth_small)

    im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                             vmax=vmax_small)
    ax[0, 1].set_title('Ground Truth (Zoom)')

    im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                             vmax=vmax_small)
    ax[1, 1].set_title('Prediction (Zoom)')

    im = ax[2, 1].pcolormesh(X_small, Y_small, np.abs(pred_small - truth_small), shading='gouraud', cmap='magma')
    fig.colorbar(im, ax=ax[2, 1])
    ax[2, 1].set_title('Absolute Error (Zoom)')

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_FILENAME)
    print(f"Saved comparison plot to {FINAL_PLOT_FILENAME}")
    plt.close(fig)