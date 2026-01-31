"""
OLD VERSION: This is outdated. upscaledFNO.py is the current one.

ZERO-SHOT SUPER-RESOLUTION TRAINING
-----------------------------------
Function:
    Trains an FNO model on Artificially Upscaled Data (2x resolution) to learn physics on a finer grid,
    then tests it on the Original (coarser) Data to prove resolution invariance.

Key Features:
    1. Synthetic Upscaling: Uses bilinear interpolation to double the resolution of the training set (221x51 -> 442x102).
    2. Zero-Shot Testing: The trained model is evaluated on the *original* low-res data without fine-tuning.
    3. Proof of Concept: Validates that FNOs learn the underlying continuous operator and are not tied to the training grid size.
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

# Differentiate file names for the Upscaled experiment
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_upscaled_model_cpu.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, "naca_upscaled_training_log.xlsx")
FINAL_PLOT_FILENAME = "naca_upscaled_zero_shot_test.png"

os.makedirs(MODEL_PATH, exist_ok=True)

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200

# Batch size
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
print("Loading raw data...")
inputX = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
# Shape: [Samples, 221, 51, 2]
input_data_raw = torch.stack([inputX, inputY], dim=-1)

output_data_raw = np.load(OUTPUT_Sigma)
if output_data_raw.ndim == 4 and output_data_raw.shape[1] > 4:
    output_data_raw = output_data_raw[:, 4]
elif output_data_raw.ndim == 3:
    pass
else:
    raise ValueError(f"Unexpected shape for OUTPUT_Sigma: {output_data_raw.shape}")
output_data_raw = torch.tensor(output_data_raw, dtype=torch.float)

# 1. Permute to [Batch, Channels, Height, Width] for Torch/FNO
# Input becomes: [N, 2, 221, 51]
# Output becomes: [N, 1, 221, 51]
x_raw = input_data_raw.permute(0, 3, 1, 2).contiguous()
y_raw = output_data_raw.unsqueeze(-1).permute(0, 3, 1, 2).contiguous()

print(f"Original Raw Shape: {x_raw.shape}")

# ---------------------------------------------------------
# UPSCALING LOGIC
# ---------------------------------------------------------
def upscale_tensor(data, scale_factor=2.0):
    """Upscales (Batch, C, H, W) data using bilinear interpolation."""
    return F.interpolate(data, scale_factor=scale_factor, mode='bilinear', align_corners=False)

print("Upscaling training data by 2x...")
# We only upscale the training portion.
# We keep a copy of raw test data for the final "Normal Resolution" test.

x_train_raw = x_raw[:ntrain]
y_train_raw = y_raw[:ntrain]

# Create Upscaled Train Data
x_train_highres = upscale_tensor(x_train_raw, scale_factor=2.0)
y_train_highres = upscale_tensor(y_train_raw, scale_factor=2.0)

# High Res Test set (for monitoring training performance)
x_test_highres = upscale_tensor(x_raw[ntrain:ntrain+ntest], scale_factor=2.0)
y_test_highres = upscale_tensor(y_raw[ntrain:ntrain+ntest], scale_factor=2.0)

# Low Res Test set (For the final requirement: test against normal resolution)
x_test_lowres = x_raw[ntrain:ntrain+ntest]
y_test_lowres = y_raw[ntrain:ntrain+ntest]

print(f"High-Res Train Shape: {x_train_highres.shape} (Used for Training)")
print(f"Low-Res Test Shape:   {x_test_lowres.shape} (Used for Final Eval)")

# --- PADDED DATASET (Dynamic Resolution) ---
class PaddedDataset(Dataset):
    def __init__(self, x, y, target_h=None, target_w=None):
        self.x = x
        self.y = y

        # Determine dimensions automatically if not provided
        current_h = x.shape[2]
        current_w = x.shape[3]

        # If target not provided, pick next reasonable multiple/power of 2
        # For Low Res (221, 51) -> We use (224, 64)
        # For High Res (442, 102) -> We use (448, 128)
        if target_h is None:
            self.target_h = 448 if current_h > 300 else 224
        else:
            self.target_h = target_h

        if target_w is None:
            self.target_w = 128 if current_w > 80 else 64
        else:
            self.target_w = target_w

        # Calculate padding needed
        self.pad_h = self.target_h - current_h
        self.pad_w = self.target_w - current_w

        if self.pad_h < 0 or self.pad_w < 0:
            raise ValueError(f"Target size ({self.target_h},{self.target_w}) smaller than input ({current_h},{current_w})")

        # Create grid for the TARGET size
        gridx = torch.tensor(np.linspace(0, 1, self.target_h), dtype=torch.float)
        gridx = gridx.reshape(1, self.target_h, 1).repeat([1, 1, self.target_w])
        gridy = torch.tensor(np.linspace(0, 1, self.target_w), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.target_w).repeat([1, self.target_h, 1])
        self.grid = torch.cat((gridx, gridy), dim=0)  # [2, H, W]

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

# 1. High Res Loaders (For Training)
# Target approx: 442->448, 102->128
train_loader = DataLoader(
    PaddedDataset(x_train_highres, y_train_highres, target_h=448, target_w=128),
    batch_size=batch_size, shuffle=True
)
test_loader_highres = DataLoader(
    PaddedDataset(x_test_highres, y_test_highres, target_h=448, target_w=128),
    batch_size=batch_size, shuffle=False
)

# 2. Low Res Loader (For Final "Normal Resolution" Test)
# Target approx: 221->224, 51->64
test_loader_lowres = DataLoader(
    PaddedDataset(x_test_lowres, y_test_lowres, target_h=224, target_w=64),
    batch_size=1, shuffle=False
)

################################################################
# Model Definition
################################################################
# FNO is resolution invariant, but we define modes based on the complexity
# we expect to capture.
model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=None, # Handling padding in Dataset
    non_linearity=torch.nn.GELU()
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(d=2, p=2, reduction='sum')

################################################################
# Training Loop (On High Res Data)
################################################################
training_history = []

print("Starting training on UPSCALED (High-Res) data...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0

    current_lr = optimizer.param_groups[0]['lr']

    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        # Standard forward/backward
        out = model(x)
        loss = loss_fn(out, y)

        loss.backward()
        optimizer.step()

        train_l2 += loss.item()

    scheduler.step()

    # Evaluate on High Res Validation set to track progress
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader_highres:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            out = model(x)
            test_l2 += loss_fn(out, y).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    epoch_time = t2 - t1

    print(f"Epoch {ep + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss (HR): {train_l2:.6f} | Test Loss (HR): {test_l2:.6f}")

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
# Final Testing against NORMAL RESOLUTION
################################################################
print("\nPerforming Zero-Shot Test on NORMAL (Low-Res) resolution...")
print("The model was trained on 442x102, but we are testing on 221x51.")

# Reload best state (or just use current)
final_model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=None,
    non_linearity=torch.nn.GELU()
).to(device)

final_model.load_state_dict(torch.load(FINAL_MODEL_FILENAME, map_location=device))
final_model.eval()

with torch.no_grad():
    # Use the Low Res Loader
    batch = next(iter(test_loader_lowres))
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    # FNO Forward pass handles the different resolution automatically
    # provided the Grid in the dataset matches the Input resolution.
    out = final_model(x)

    # Calculate one-shot loss on this low-res sample
    sample_loss = loss_fn(out, y).item()
    print(f"Sample Zero-Shot Loss on Low-Res: {sample_loss:.6f}")

    # Unpad for visualization: Slice back to [221, 51]
    # Note: We are using the Low-Res dimensions here
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
    ax[0, 0].set_title(f'Ground Truth (Normal Res: {valid_h}x{valid_w})')

    im = ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title(f'Prediction (Model trained on 2x Res)')

    im = ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title('Difference')

    # Zoom logic
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