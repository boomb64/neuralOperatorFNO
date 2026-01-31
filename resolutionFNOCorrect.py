"""
FNO MULTI-RESOLUTION SOLVER (CPU Optimized)
-------------------------------------------
Function:
    The "Worker" script that trains a single FNO model at a specific resolution (r=2, 4, 8, etc.).
    It is designed to be called repeatedly by the 'resolutionRUNNER' script.

Key Features:
    1. Variable Resolution: accepts a `--subsample_r` argument to train on coarse data (simulating sparse sensors).
    2. Zero-Shot Super-Resolution: After training on coarse data, it immediately evaluates the model on the
       *Full Resolution* test set to measure how well it generalizes to finer grids.
    3. Gold Standard Comparison: Loads the baseline (r=1) model to calculate the exact "Accuracy Gap"
       introduced by downsampling the training data.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import math
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F

# --- neuraloperator imports ---
from neuralop.models import FNO
from neuralop.losses import LpLoss

# ---------------------------------------
# Arguments
# ---------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--subsample_r', type=int, default=1,
                    help='Subsampling rate. 1=Full, 2=Half, 4=Quarter, etc.')
parser.add_argument('--gold_model_path', type=str, default=None,
                    help='Path to r=1 model. Defaults to ./model/naca_final_model_cpu.pth')
args = parser.parse_args()

r = args.subsample_r
print(f"--- Running on CPU with Subsample Rate (r) = {r} ---")

# ---------------------------------------
# Configuration
# ---------------------------------------
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cpu")
print(f"Forcing device to: {device}")

BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")
PLOTS_PATH = os.path.join(BASE_PATH, "results")

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

# Dynamic filenames
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, f"naca_fno_r{r}_cpu.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, f"naca_log_r{r}_cpu.xlsx")
FINAL_PLOT_FILENAME = os.path.join(PLOTS_PATH, f"naca_plot_r{r}_cpu.png")

# Gold Standard Path
if args.gold_model_path:
    GOLD_MODEL_FILENAME = args.gold_model_path
else:
    # Default to the name defined in your original snippet
    GOLD_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_final_model_cpu.pth")

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200
batch_size = 32
learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5

width = 32
base_modes = 12

# Subsampling Grid Calc
r1, r2 = r, r
s1_full, s2_full = 221, 51
s1 = int(((s1_full - 1) / r1) + 1)
s2 = int(((s2_full - 1) / r2) + 1)

print(f"Full Grid: {s1_full}x{s2_full} -> Training Grid: {s1}x{s2}")

# Dynamic Modes (Prevent modes > size/2)
modes_x = min(base_modes * 2, s1 // 2)
modes_y = min(base_modes, s2 // 2)
print(f"Using Modes: ({modes_x}, {modes_y})")

################################################################
# Data Loading & Preprocessing
################################################################
print("Loading data...")
# Load Full Res
inputX_full = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY_full = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
input_data_full = torch.stack([inputX_full, inputY_full], dim=-1) # [N, 221, 51, 2]

output_data_full_np = np.load(OUTPUT_Sigma)
if output_data_full_np.ndim == 4 and output_data_full_np.shape[1] > 4:
    output_data_full = torch.tensor(output_data_full_np[:, 4], dtype=torch.float)
elif output_data_full_np.ndim == 3:
    output_data_full = torch.tensor(output_data_full_np, dtype=torch.float)
else:
    raise ValueError("Unexpected shape for output")

# 1. Prepare Subsampled Training Data
x_train = input_data_full[:ntrain, ::r1, ::r2]
y_train = output_data_full[:ntrain, ::r1, ::r2].unsqueeze(-1)

# 2. Prepare Subsampled Test Data (For Validation)
x_test = input_data_full[ntrain:ntrain+ntest, ::r1, ::r2]
y_test = output_data_full[ntrain:ntrain+ntest, ::r1, ::r2].unsqueeze(-1)

# 3. Prepare FULL Res Test Data (For Zero-Shot Comparison)
x_test_full = input_data_full[ntrain:ntrain+ntest]
y_test_full = output_data_full[ntrain:ntrain+ntest].unsqueeze(-1)

# Permute all to [Batch, Channels, H, W]
x_train = x_train.permute(0, 3, 1, 2).contiguous()
x_test = x_test.permute(0, 3, 1, 2).contiguous()
x_test_full = x_test_full.permute(0, 3, 1, 2).contiguous()

y_train = y_train.permute(0, 3, 1, 2).contiguous()
y_test = y_test.permute(0, 3, 1, 2).contiguous()
y_test_full = y_test_full.permute(0, 3, 1, 2).contiguous()

print(f"Train Input Shape: {x_train.shape}")

# --- PADDED DATASET (Dynamic) ---
class PaddedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Find nearest efficient padding (multiples of 4 or 8 are good for FFT)
        # We pad slightly larger than the input size
        h, w = x.shape[2], x.shape[3]

        # Simple logic: Pad to next multiple of 8 if not already
        self.target_h = math.ceil(h / 8) * 8
        self.target_w = math.ceil(w / 8) * 8

        # Ensure it's at least slightly padded if they match exactly (optional, but safe)
        if self.target_h == h: self.target_h += 8
        if self.target_w == w: self.target_w += 8

        # Calculate padding needed
        self.pad_h = self.target_h - h
        self.pad_w = self.target_w - w

        # Create grid for the TARGET size
        gridx = torch.tensor(np.linspace(0, 1, self.target_h), dtype=torch.float)
        gridx = gridx.reshape(1, self.target_h, 1).repeat([1, 1, self.target_w])
        gridy = torch.tensor(np.linspace(0, 1, self.target_w), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.target_w).repeat([1, self.target_h, 1])
        self.grid = torch.cat((gridx, gridy), dim=0)  # [2, H_pad, W_pad]

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

# Create Datasets
train_ds = PaddedDataset(x_train, y_train)
test_ds = PaddedDataset(x_test, y_test)
test_full_ds = PaddedDataset(x_test_full, y_test_full)

# Create Loaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
test_full_loader = DataLoader(test_full_ds, batch_size=batch_size, shuffle=False)
plot_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

################################################################
# Model Definition
################################################################
model = FNO(
    n_modes=(modes_x, modes_y),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=None, # Handled by dataset
    non_linearity=torch.nn.GELU()
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(d=2, p=2, reduction='sum')

################################################################
# Training Loop
################################################################
training_history = []
print(f"Starting training on CPU (r={r})...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0
    current_lr = optimizer.param_groups[0]['lr']

    for batch in train_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
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

# Save Results
try:
    pd.DataFrame(training_history).to_excel(EXCEL_FILENAME, index=False)
except Exception as e:
    print(f"Error saving history: {e}")

torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Saved model to {FINAL_MODEL_FILENAME}")

################################################################
# 1. Zero-Shot Evaluation (Current Model on Full Res Data)
################################################################
print("\n--- Running Zero-Shot Super-Resolution Check ---")
model.eval()
zero_shot_loss = 0.0
with torch.no_grad():
    for batch in test_full_loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        # FNO is resolution invariant; it accepts the larger Full Res grid
        out = model(x)
        zero_shot_loss += loss_fn(out, y).item()

zero_shot_loss /= ntest
print(f"Zero-Shot Loss (Model r={r} on Full Data): {zero_shot_loss:.6f}")

################################################################
# 2. Gold Standard Comparison
################################################################
print("\n--- Comparing against Gold Standard (r=1) Model ---")
if os.path.exists(GOLD_MODEL_FILENAME):
    # Initialize a model with Full Res architecture (12, 12 modes)
    # Assuming Gold Standard was trained with defaults
    gold_modes_x = base_modes * 2
    gold_modes_y = base_modes

    gold_model = FNO(
        n_modes=(gold_modes_x, gold_modes_y),
        hidden_channels=width,
        in_channels=4,
        out_channels=1,
        n_layers=4,
        domain_padding=None,
        non_linearity=torch.nn.GELU()
    ).to(device)

    # Safe deserialization
    from neuralop.layers.spectral_convolution import SpectralConv
    torch.serialization.add_safe_globals([torch.nn.modules.activation.GELU, SpectralConv])

    try:
        gold_model.load_state_dict(torch.load(GOLD_MODEL_FILENAME, map_location=device))
        gold_model.eval()

        gold_loss = 0.0
        with torch.no_grad():
            for batch in test_full_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                out = gold_model(x)
                gold_loss += loss_fn(out, y).item()

        gold_loss /= ntest
        print(f"Gold Standard Loss (Model r=1 on Full Data): {gold_loss:.6f}")
        print(f"Accuracy Gap: {zero_shot_loss - gold_loss:.6f}")

    except Exception as e:
        print(f"Error loading gold standard model: {e}")
        print("Ensure the architecture (width/modes) matches exactly.")
else:
    print(f"Gold model not found at {GOLD_MODEL_FILENAME}. Run r=1 first.")

################################################################
# Plotting (Current Resolution)
################################################################
print("\nGenerating comparison plot...")
with torch.no_grad():
    batch = next(iter(plot_loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    out = model(x)

    # Unpad for visualization
    # We need to know the valid unpadded size for this resolution
    valid_h = s1
    valid_w = s2

    X = x[0, 0, :valid_h, :valid_w].numpy()
    Y = x[0, 1, :valid_h, :valid_w].numpy()
    truth = y[0, 0, :valid_h, :valid_w].numpy()
    pred = out[0, 0, :valid_h, :valid_w].numpy()

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    im = ax[0, 0].pcolormesh(X, Y, truth, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0, 0].set_title(f'Ground Truth (r={r})')
    im = ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1, 0].set_title(f'Prediction (r={r})')
    im = ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud', cmap='coolwarm')
    ax[2, 0].set_title('Difference')

    # Zoom
    nx, ny = 20 // r, 10 // r # Scale zoom window by r
    # Safe slice
    if X.shape[0] > 2*nx and X.shape[1] > ny and nx > 0:
        X_s, Y_s = X[nx:-nx, :ny], Y[nx:-nx, :ny]
        t_s, p_s = truth[nx:-nx, :ny], pred[nx:-nx, :ny]

        im = ax[0, 1].pcolormesh(X_s, Y_s, t_s, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        ax[0, 1].set_title('Ground Truth Zoom')
        im = ax[1, 1].pcolormesh(X_s, Y_s, p_s, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
        ax[1, 1].set_title('Prediction Zoom')
        im = ax[2, 1].pcolormesh(X_s, Y_s, np.abs(p_s - t_s), shading='gouraud', cmap='magma')
        ax[2, 1].set_title('Absolute Error Zoom')

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_FILENAME)
    print(f"Saved plot to {FINAL_PLOT_FILENAME}")