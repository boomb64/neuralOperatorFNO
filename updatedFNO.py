"""
MANUAL FNO IMPLEMENTATION (RTX 5070 - Float32)
------------------------------------------------------
Fixes:
1. Removed AMP (Mixed Precision): Fixes the cuFFT power-of-two error.
2. Batch Size = 4: Fits in 12GB VRAM (consumes ~2GB).
3. TF32 Enabled: Accelerates Float32 math on your 50-series card.
"""

import os
import torch
import numpy as np
import matplotlib
# Use non-interactive backend for training
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import time
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

# --- neuraloperator imports ---
from neuralop.models import FNO
from neuralop.losses import LpLoss

# ---------------------------------------
# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# Enable TF32 (Great for RTX 5070)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

################################################################
# Configurations
################################################################
BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")

# Filenames
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "fno_model.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, "fno_training_log_manual.xlsx")
FINAL_PLOT_FILENAME = "fno_final_comparison_manual.png"

os.makedirs(MODEL_PATH, exist_ok=True)

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200

# Batch Size 4 fits in Float32 on 12GB VRAM
batch_size = 4

learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5

modes = 12
width = 32

r1, r2 = 1, 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

################################################################
# Data Loading & Preprocessing
################################################################
print("Loading data...")
inputX = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
input_data = torch.stack([inputX, inputY], dim=-1)

output_data = np.load(OUTPUT_Sigma)
if output_data.ndim == 4 and output_data.shape[1] > 4:
    output_data = output_data[:, 4]
elif output_data.ndim == 3:
    pass
else:
    raise ValueError(f"Unexpected shape for OUTPUT_Sigma: {output_data.shape}")
output_data = torch.tensor(output_data, dtype=torch.float)

print(f"Original input shape: {input_data.shape}, Output shape: {output_data.shape}")

x_train = input_data[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input_data[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]

x_train = x_train.permute(0, 3, 1, 2).contiguous()
x_test = x_test.permute(0, 3, 1, 2).contiguous()

y_train = output_data[:ntrain, ::r1, ::r2][:, :s1, :s2].unsqueeze(1)
y_test = output_data[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2].unsqueeze(1)

class DictDataset(Dataset):
    def __init__(self, x, y, grid_h, grid_w):
        self.x = x
        self.y = y
        gridx = torch.tensor(np.linspace(0, 1, grid_h), dtype=torch.float)
        gridx = gridx.reshape(1, grid_h, 1).repeat([1, 1, grid_w])
        gridy = torch.tensor(np.linspace(0, 1, grid_w), dtype=torch.float)
        gridy = gridy.reshape(1, 1, grid_w).repeat([1, grid_h, 1])
        self.grid = torch.cat((gridx, gridy), dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_deformed = self.x[idx]
        x_combined = torch.cat((x_deformed, self.grid), dim=0)
        return {"x": x_combined, "y": self.y[idx]}

# Pin memory = True for GPU
train_loader = DataLoader(DictDataset(x_train, y_train, s1, s2), batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(DictDataset(x_test, y_test, s1, s2), batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader2 = DataLoader(DictDataset(x_test, y_test, s1, s2), batch_size=1, shuffle=False)

################################################################
# Model Definition
################################################################
device = torch.device("cuda")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

# Clean memory
torch.cuda.empty_cache()

model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=[8, 8],
    non_linearity=torch.nn.GELU()
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(d=2, p=2, reduction='sum')

# REMOVED: scaler (Not using AMP)

################################################################
# Training Loop
################################################################
training_history = []
print("Starting training on GPU (Float32)...")

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0
    current_lr = optimizer.param_groups[0]['lr']

    for batch_idx, batch in enumerate(train_loader):
        # Non-blocking transfer speeds up data loading
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Standard Forward Pass (No Autocast)
        out = model(x)
        loss = loss_fn(out, y)

        # Standard Backward Pass
        loss.backward()
        optimizer.step()

        train_l2 += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"   [Epoch {ep+1}] Processing batch {batch_idx+1}...")

    scheduler.step()

    # Evaluation loop
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            # Standard Forward Pass
            out = model(x)
            test_l2 += loss_fn(out, y).item()

    train_l2 /= ntrain
    test_l2 /= ntest
    t2 = default_timer()
    epoch_time = t2 - t1

    print(f"Epoch {ep + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_l2:.6f} | Test Loss: {test_l2:.6f}")

    training_history.append({
        'epoch': ep + 1,
        'time_s': epoch_time,
        'train_loss': train_l2,
        'test_loss': test_l2,
        'learning_rate': current_lr
    })

################################################################
# Save History & Model
################################################################
print(f"Saving training history to {EXCEL_FILENAME}...")
try:
    df_history = pd.DataFrame(training_history)
    df_history.to_excel(EXCEL_FILENAME, index=False)
    print("History saved successfully.")
except Exception as e:
    print(f"Error saving history to Excel: {e}")

torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Saved final model to {FINAL_MODEL_FILENAME}")

################################################################
# Final Evaluation
################################################################
print("Generating final comparison plot...")

final_model = FNO(
    n_modes=(modes * 2, modes),
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
    n_layers=4,
    domain_padding=[8, 8],
    non_linearity=torch.nn.GELU()
).to(device)

# Load with weights_only=False fix
final_model.load_state_dict(torch.load(FINAL_MODEL_FILENAME, map_location=device, weights_only=False))
final_model.eval()

with torch.no_grad():
    batch = next(iter(test_loader2))
    x = batch["x"].to(device)
    y = batch["y"].to(device)
    out = final_model(x)

    X = x[0, 0].cpu().numpy()
    Y = x[0, 1].cpu().numpy()
    truth = y[0].squeeze().cpu().numpy()
    pred = out[0].squeeze().cpu().numpy()

    # Zoom logic
    nx = 40 // r1
    ny = 20 // r2
    if X.shape[0] > 2 * nx and X.shape[1] > ny:
        X_small = X[nx:-nx, :ny]
        Y_small = Y[nx:-nx, :ny]
        truth_small = truth[nx:-nx, :ny]
        pred_small = pred[nx:-nx, :ny]
    else:
        X_small, Y_small, truth_small, pred_small = X, Y, truth, pred

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    im = ax[0, 0].pcolormesh(X, Y, truth, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title('Ground Truth (Full)')

    im = ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title('Prediction (Full)')

    im = ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title('Difference (Full)')

    vmin_small, vmax_small = np.min(truth_small), np.max(truth_small)
    im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small, vmax=vmax_small)
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title('Ground Truth (Zoom)')

    im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small, vmax=vmax_small)
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title('Prediction (Zoom)')

    im = ax[2, 1].pcolormesh(X_small, Y_small, np.abs(pred_small - truth_small), shading='gouraud', cmap='magma')
    fig.colorbar(im, ax=ax[2, 1])
    ax[2, 1].set_title('Absolute Error (Zoom)')

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_FILENAME)
    print(f"Saved comparison plot to {FINAL_PLOT_FILENAME}")
    plt.close(fig)