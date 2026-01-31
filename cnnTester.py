"""
THIS IS AN OLDER FILE, NOT USED CURRENTLY. Use updatedCNN.py instead

CNN BASELINE IMPLEMENTATION (NACA)
----------------------------------
Function:
    Trains a standard Convolutional Neural Network (CNN) on the NACA airfoil dataset.
    It maps Airfoil Geometry -> Flow Field, serving as a baseline comparison for FNO models.

Key Features:
    1. Architecture: Simple 6-layer CNN with GELU activations (no Fourier layers).
    2. Loss Metric: Uses MSELoss (Sum reduction) for standard regression training.
    3. Benchmarking: Saves training history and error metrics to allow direct comparison
       with the FNO results to quantify the benefit of operator learning.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time
import pandas as pd
import torch.nn as nn  # --- ADDED ---
import torch.nn.functional as F  # --- ADDED ---

# --- REMOVED neuraloperator imports ---

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# Configurations
################################################################
BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")

# --- CHANGED Filenames to avoid overwriting FNO results ---
FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_final_model_cnn.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, "naca_training_log_cnn.xlsx")
FINAL_PLOT_FILENAME = "naca_final_comparison_cnn.png"

os.makedirs(MODEL_PATH, exist_ok=True)

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 1000
ntest = 200
batch_size = 20
learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5

# --- CHANGED ---
# modes = 12  # FNO-specific, removed
width = 32  # CNN hidden channels, kept from original

r1, r2 = 1, 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

################################################################
# Data Loading & Preprocessing
################################################################
inputX = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
input = torch.stack([inputX, inputY], dim=-1)

output = np.load(OUTPUT_Sigma)
if output.ndim == 4 and output.shape[1] > 4:
    output = output[:, 4]
elif output.ndim == 3:
    pass
else:
    raise ValueError(f"Unexpected shape for OUTPUT_Sigma: {output.shape}")
output = torch.tensor(output, dtype=torch.float)

print(f"Original input shape: {input.shape}, Original output shape: {output.shape}")

x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
x_test = x_test.permute(0, 3, 1, 2).contiguous()    # [B, C, H, W]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2].unsqueeze(1)
y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2].unsqueeze(1)

print(f"Processed x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"Processed x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# --- Dataset class returning dicts (Unchanged) ---
class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


train_loader = DataLoader(DictDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(DictDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
test_loader2 = DataLoader(DictDataset(x_test, y_test), batch_size=1, shuffle=False)

################################################################
# Model Definition
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- ADDED SimpleCNN Model ---
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, width=32):
        super(SimpleCNN, self).__init__()
        # Use kernel_size=5 and padding=2 to preserve spatial dimensions
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(width * 4, width * 2, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(width * 2, width, kernel_size=5, padding=2)
        # 1x1 conv to map to output channels
        self.conv6 = nn.Conv2d(width, out_channels, kernel_size=1)

    def forward(self, x):
        # Using GELU activation to match the FNO's non_linearity
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = F.gelu(self.conv4(x))
        x = F.gelu(self.conv5(x))
        x = self.conv6(x)
        return x
# --- END ADDED ---


# --- CHANGED Model Instantiation ---
model = SimpleCNN(
    in_channels=2,
    out_channels=1,
    width=width
).to(device)

print(f"Using SimpleCNN model. Total params: {sum(p.numel() for p in model.parameters())}")
# --- END CHANGED ---

################################################################
# Training Setup
################################################################
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# --- CHANGED Loss Function ---
loss_fn = nn.MSELoss(reduction='sum')
# --- END CHANGED ---

################################################################
# Custom Trainer with progress printing
################################################################
# --- CHANGED: Removed inheritance from neuralop.Trainer ---
class VerboseTrainer:
    def __init__(self, model, device, n_epochs, optimizer):
        self.model = model
        self.device = device
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.history = []  # Initialize history list

    def train_one_batch(self, idx, sample, training_loss):
        x = sample["x"].to(self.device)
        y = sample["y"].to(self.device)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = training_loss(out, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, loader, loss_fn):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                out = self.model(x)
                total_loss += loss_fn(out, y).item()
        self.model.train()
        return total_loss / len(loader.dataset) # Get loss per sample

    def train(self, train_loader, test_loaders=None, training_loss=None, eval_losses=None, scheduler=None):
        for epoch in range(self.n_epochs):
            start = time.time()

            if scheduler:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            # Train
            total_train_loss = 0
            for idx, batch in enumerate(train_loader):
                total_train_loss += self.train_one_batch(idx, batch, training_loss)
            train_loss = total_train_loss / len(train_loader.dataset) # Get loss per sample

            # Evaluate
            test_losses = {}
            if test_loaders:
                for name, loader in test_loaders.items():
                    # --- CHANGED: Updated loss key from "LpLoss" to "MSE" ---
                    test_losses[name] = self.evaluate(loader, eval_losses["MSE"])

            # Scheduler step
            if scheduler:
                scheduler.step()

            end = time.time()

            epoch_time = end - start
            test_loss_value = test_losses.get('test', float('nan')) if test_loaders else float('nan')

            epoch_data = {
                'epoch': epoch + 1,
                'time_s': epoch_time,
                'train_loss': train_loss,
                'test_loss': test_loss_value,
                'learning_rate': current_lr
            }
            self.history.append(epoch_data)

            msg = f"Epoch {epoch+1}/{self.n_epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f}"
            for name, l in test_losses.items():
                msg += f" | {name} Loss: {l:.6f}"
            print(msg)


################################################################
# Training
################################################################
trainer = VerboseTrainer(model=model, device=device, n_epochs=epochs, optimizer=optimizer)

# --- CHANGED: Updated eval_losses key ---
trainer.train(
    train_loader=train_loader,
    test_loaders={"test": test_loader},
    training_loss=loss_fn,
    eval_losses={"MSE": loss_fn}, # Was "LpLoss"
    scheduler=scheduler
)
# --- END CHANGED ---

# --- Save history to Excel (Unchanged) ---
print(f"Saving training history to {EXCEL_FILENAME}...")
try:
    df_history = pd.DataFrame(trainer.history)
    df_history.to_excel(EXCEL_FILENAME, index=False)
    print("History saved successfully.")
except Exception as e:
    print(f"Error saving history to Excel: {e}")


torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Saved final model to {FINAL_MODEL_FILENAME}")

################################################################
# Final Evaluation & Plot
################################################################
print("Generating final comparison plot...")

# --- CHANGED: Recreate CNN model architecture ---
final_model = SimpleCNN(
    in_channels=2,
    out_channels=1,
    width=width
).to(device)
# --- END CHANGED ---

# --- REMOVED: FNO-specific deserialization helpers ---
# from neuralop.layers.spectral_convolution import SpectralConv
# torch.serialization.add_safe_globals([torch.nn.modules.activation.GELU, SpectralConv])
# --- END REMOVED ---

# Load weights
final_model.load_state_dict(torch.load(FINAL_MODEL_FILENAME, map_location=device))
final_model.eval()

# Single-batch inference for plotting (Unchanged)
with torch.no_grad():
    batch = next(iter(test_loader2))
    x, y = batch["x"].to(device), batch["y"].to(device)
    out = final_model(x)

    X = x[0, 0].cpu().numpy()
    Y = x[0, 1].cpu().numpy()
    truth = y[0].squeeze().cpu().numpy()
    pred = out[0].squeeze().cpu().numpy()

    # Optional zoomed-in region
    nx, ny = 40 // r1, 20 // r2
    if X.shape[0] > 2 * nx and X.shape[1] > ny:
        X_small, Y_small = X[nx:-nx, :ny], Y[nx:-nx, :ny]
        truth_small, pred_small = truth[nx:-nx, :ny], pred[nx:-nx, :ny]
    else:
        X_small, Y_small, truth_small, pred_small = X, Y, truth, pred

    # Plotting (Unchanged)
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    # Full view
    im = ax[0, 0].pcolormesh(X, Y, truth, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title('Ground Truth (Full)')

    im = ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title('Prediction (Full)')

    im = ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title('Difference (Full)')

    # Zoomed-in view
    vmin_small, vmax_small = np.min(truth_small), np.max(truth_small)

    im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title('Ground Truth (Zoom)')

    im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title('Prediction (Zoom)')

    im = ax[2, 1].pcolormesh(X_small, Y_small, pred_small - truth_small, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 1])
    ax[2, 1].set_title('Difference (Zoom)')

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_FILENAME)
    print(f"Saved comparison plot to {FINAL_PLOT_FILENAME}")
    # plt.show()