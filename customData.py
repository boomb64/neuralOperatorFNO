"""
OLD FILE: was used early to debug

CUSTOM DATASET FNO TRAINER
--------------------------
Function:
    A simplified FNO training script specifically designed for small, custom datasets
    (like your manually processed X-Z slice data).

Key Features:
    1. Small-Scale Config: Default parameters (batch_size=4, ntrain=6) are tuned for rapid
       prototyping on tiny datasets to verify pipeline correctness before scaling up.
    2. Input Flexibility: Hardcoded to handle X-Z coordinate inputs (common in 2D airfoil slices)
       instead of the standard X-Y.
    3. Sanity Checks: Includes explicit print statements for data ranges (min/max/mean) to
       catch normalization errors early.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import time

# --- neuraloperator imports (v2.0.0) ---
from neuralop.models import FNO
from neuralop.losses import LpLoss
from neuralop.training import Trainer
# ---------------------------------------

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
DATA_PATH = os.path.join(BASE_PATH, "data", "tecplot_processed_simple_XZ")

FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "naca_final_model_neuralop.pth")
FINAL_PLOT_FILENAME = "naca_final_comparison_neuralop.png"

os.makedirs(MODEL_PATH, exist_ok=True)

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Z.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

ntrain = 6
ntest = 1
batch_size = 4
learning_rate = 0.001
epochs = 50
step_size = 100
gamma = 0.5

modes = 12
width = 32

r1, r2 = 1, 1
s1 = 200
s2 = 200

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
print(f"Output data sanity check: min={output.min()}, max={output.max()}, mean={output.mean()}")
# --- Dataset class returning dicts ---
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

model = FNO(
    n_modes=(modes, modes),
    hidden_channels=width,
    in_channels=2,
    out_channels=1,
    n_layers=4,
    domain_padding=None,
    non_linearity=torch.nn.GELU()
).to(device)

print(f"Using neuraloperator FNO model. Total params: {sum(p.numel() for p in model.parameters())}")

################################################################
# Training Setup
################################################################
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
loss_fn = LpLoss(d=2, p=2, reduction='sum')  # Sum, not mean

################################################################
# Custom Trainer with progress printing
################################################################
class VerboseTrainer(Trainer):
    def __init__(self, model, device, n_epochs, optimizer):
        self.model = model
        self.device = device
        self.n_epochs = n_epochs
        self.optimizer = optimizer

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
        return total_loss / len(loader)

    def train(self, train_loader, test_loaders=None, training_loss=None, eval_losses=None, scheduler=None):
        for epoch in range(self.n_epochs):
            start = time.time()
            # Train
            total_train_loss = 0
            for idx, batch in enumerate(train_loader):
                total_train_loss += self.train_one_batch(idx, batch, training_loss)
            train_loss = total_train_loss / len(train_loader)

            # Evaluate
            test_losses = {}
            if test_loaders:
                for name, loader in test_loaders.items():
                    test_losses[name] = self.evaluate(loader, eval_losses["LpLoss"])

            # Scheduler step
            if scheduler:
                scheduler.step()

            end = time.time()
            msg = f"Epoch {epoch+1}/{self.n_epochs} | Time: {end - start:.2f}s | Train Loss: {train_loss:.6f}"
            for name, l in test_losses.items():
                msg += f" | {name} Loss: {l:.6f}"
            print(msg)


################################################################
# Training
################################################################
trainer = VerboseTrainer(model=model, device=device, n_epochs=epochs, optimizer=optimizer)
trainer.train(
    train_loader=train_loader,
    test_loaders={"test": test_loader},
    training_loss=loss_fn,
    eval_losses={"LpLoss": loss_fn},
    scheduler=scheduler
)

torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Saved final model to {FINAL_MODEL_FILENAME}")

################################################################
# Final Evaluation & Plot (Fixed for PyTorch 2.6+)
################################################################
print("Generating final comparison plot...")

# Recreate model architecture
final_model = FNO(
    n_modes=(modes, modes),
    hidden_channels=width,
    in_channels=2,
    out_channels=1,
    n_layers=4,
    domain_padding=None,
    non_linearity=torch.nn.GELU()
).to(device)

# Allow GELU and SpectralConv for safe deserialization
from neuralop.layers.spectral_convolution import SpectralConv
torch.serialization.add_safe_globals([torch.nn.modules.activation.GELU, SpectralConv])

# Load weights safely
final_model.load_state_dict(torch.load(FINAL_MODEL_FILENAME, map_location=device))
final_model.eval()

# Single-batch inference for plotting
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

    # Plotting
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    # Full view
    im = ax[0, 0].pcolormesh(X, Y, truth, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[0, 0]); ax[0, 0].set_title('Ground Truth (Full)')
    im = ax[1, 0].pcolormesh(X, Y, pred, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0]); ax[1, 0].set_title('Prediction (Full)')
    im = ax[2, 0].pcolormesh(X, Y, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0]); ax[2, 0].set_title('Difference (Full)')

    # Zoomed-in view
    vmin_small, vmax_small = np.min(truth_small), np.max(truth_small)
    im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small, vmax=vmax_small)
    fig.colorbar(im, ax=ax[0, 1]); ax[0, 1].set_title('Ground Truth (Zoom)')
    im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small, vmax=vmax_small)
    fig.colorbar(im, ax=ax[1, 1]); ax[1, 1].set_title('Prediction (Zoom)')
    im = ax[2, 1].pcolormesh(X_small, Y_small, pred_small - truth_small, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 1]); ax[2, 1].set_title('Difference (Zoom)')

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_FILENAME)
    print(f"Saved comparison plot to {FINAL_PLOT_FILENAME}")
    plt.show()