"""
OUTDATED: All functionality is contained within train_naca_upscaled

ZERO-SHOT SUPER-RESOLUTION EVALUATOR
------------------------------------
Function:
    Loads a pre-trained "Upscaled" FNO model (trained on high-resolution data) and evaluates it
    on the original, standard-resolution NACA dataset.

Key Features:
    1. Resolution Invariance Test: Demonstrates the FNO's unique ability to generalize across
       different grid discretizations (Zero-Shot Super-Resolution) without retraining.
    2. Cross-Resolution Inference: Takes a model with weights learned from dense grids and
       applies them to coarse grids to check for physical consistency.
    3. Visualization: Generates side-by-side plots of Ground Truth vs. Prediction to visually
       verify if the physics were preserved across resolutions.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from neuralop.models import FNO
from neuralop.losses import LpLoss

# ---------------------------------------
# Configuration
# ---------------------------------------
# Force CPU
device = torch.device("cpu")
print(f"Forcing device to: {device}")

BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")

# Point to the file you just saved
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_upscaled_model_cpu.pth")
FINAL_PLOT_FILENAME = "naca_upscaled_zero_shot_test_RECOVERED.png"

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200
modes = 12
width = 32

################################################################
# Data Loading (Low Res Only)
################################################################
print("Loading raw data for testing...")
inputX = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
input_data_raw = torch.stack([inputX, inputY], dim=-1)

output_data_raw = np.load(OUTPUT_Sigma)
if output_data_raw.ndim == 4 and output_data_raw.shape[1] > 4:
    output_data_raw = output_data_raw[:, 4]
elif output_data_raw.ndim == 3:
    pass
else:
    raise ValueError(f"Unexpected shape: {output_data_raw.shape}")
output_data_raw = torch.tensor(output_data_raw, dtype=torch.float)

# Permute to [Batch, Channels, Height, Width]
x_raw = input_data_raw.permute(0, 3, 1, 2).contiguous()
y_raw = output_data_raw.unsqueeze(-1).permute(0, 3, 1, 2).contiguous()

# Extract ONLY the test set (Low Res)
x_test_lowres = x_raw[ntrain:ntrain + ntest]
y_test_lowres = y_raw[ntrain:ntrain + ntest]

print(f"Low-Res Test Shape: {x_test_lowres.shape}")


# --- PADDED DATASET CLASS (Required to reconstruct data) ---
class PaddedDataset(Dataset):
    def __init__(self, x, y, target_h=None, target_w=None):
        self.x = x
        self.y = y

        current_h = x.shape[2]
        current_w = x.shape[3]

        if target_h is None:
            self.target_h = 448 if current_h > 300 else 224
        else:
            self.target_h = target_h

        if target_w is None:
            self.target_w = 128 if current_w > 80 else 64
        else:
            self.target_w = target_w

        self.pad_h = self.target_h - current_h
        self.pad_w = self.target_w - current_w

        # Create grid
        gridx = torch.tensor(np.linspace(0, 1, self.target_h), dtype=torch.float)
        gridx = gridx.reshape(1, self.target_h, 1).repeat([1, 1, self.target_w])
        gridy = torch.tensor(np.linspace(0, 1, self.target_w), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.target_w).repeat([1, self.target_h, 1])
        self.grid = torch.cat((gridx, gridy), dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_raw = self.x[idx]
        y_raw = self.y[idx]
        x_padded = F.pad(x_raw, (0, self.pad_w, 0, self.pad_h))
        y_padded = F.pad(y_raw, (0, self.pad_w, 0, self.pad_h))
        x_combined = torch.cat((x_padded, self.grid), dim=0)
        return {"x": x_combined, "y": y_padded}


# Loader for Low Res (Targeting 224x64)
test_loader_lowres = DataLoader(
    PaddedDataset(x_test_lowres, y_test_lowres, target_h=224, target_w=64),
    batch_size=1, shuffle=False
)

################################################################
# Model Definition & LOADING THE FIX
################################################################
model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=None,
    non_linearity=torch.nn.GELU()
).to(device)

print(f"Loading model from {FINAL_MODEL_FILENAME}...")

# --- THE FIX IS HERE: weights_only=False ---
try:
    state_dict = torch.load(FINAL_MODEL_FILENAME, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except Exception as e:
    print("------------------------------------------------")
    print("CRITICAL ERROR LOADING MODEL:")
    print(e)
    print("------------------------------------------------")
    exit()

model.eval()
loss_fn = LpLoss(d=2, p=2, reduction='sum')

################################################################
# Evaluation & Plotting
################################################################
print("Generating comparison plot...")

with torch.no_grad():
    batch = next(iter(test_loader_lowres))
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    # Forward pass
    out = model(x)

    sample_loss = loss_fn(out, y).item()
    print(f"Sample Zero-Shot Loss on Low-Res: {sample_loss:.6f}")

    # Unpad back to 221x51
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