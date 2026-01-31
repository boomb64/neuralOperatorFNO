"""
CNN BASELINE WITH LPLOSS & BATCHNORM
------------------------------------
Function:
    Trains a standard Convolutional Neural Network (CNN) as a baseline comparison against FNO models.
    It maps Airfoil Geometry -> Flow Field (Pressure/Velocity).

Key Features:
    1. Metric Parity: Uses LpLoss (Relative L2 Error) instead of MSE. This outputs error values (e.g., 0.05)
       that are directly comparable to FNO results, representing a percentage error (5%).
    2. Stability Fix: Includes Batch Normalization layers to prevent gradient explosion, a common issue
       when training simple CNNs on this physics dataset.
    3. Position Awareness: Appends coordinate grids (x, y) to inputs internally to help the CNN learn spatial dependencies.
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
import torch.nn as nn
import torch.nn.functional as F

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

FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "cnn_model.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, "cnn_training_log.xlsx")
FINAL_PLOT_FILENAME = "cnn_comparison_plot.png"

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

width = 32

r1, r2 = 1, 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

################################################################
# Data Loading
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
x_train = x_train.permute(0, 3, 1, 2).contiguous()  # [B, C=2, H, W]
x_test = x_test.permute(0, 3, 1, 2).contiguous()    # [B, C=2, H, W]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2].unsqueeze(1)
y_test = output[ntrain:ntrain + ntest, ::r1, ::r2][:, :s1, :s2].unsqueeze(1)

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
# Loss Function (LpLoss - The "Comparable" Metric)
################################################################
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size()[0]
        # Flatten to [Batch, Pixels] for norm calculation
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

################################################################
# Model Definition (CNN with BatchNorm)
################################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self, width=32):
        super(SimpleCNN, self).__init__()

        # 4 input channels (2 data + 2 grid)
        self.conv1 = nn.Conv2d(4, width, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(width) # Added BatchNorm

        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(width * 2) # Added BatchNorm

        self.conv3 = nn.Conv2d(width * 2, width * 4, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(width * 4) # Added BatchNorm

        self.conv4 = nn.Conv2d(width * 4, width * 2, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(width * 2) # Added BatchNorm

        self.conv5 = nn.Conv2d(width * 2, width, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(width) # Added BatchNorm

        self.conv6 = nn.Conv2d(width, 1, kernel_size=1)

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, 1, 1, size_y])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, 1, size_y).repeat([batchsize, 1, size_x, 1])
        return torch.cat((gridx, gridy), dim=1).to(device)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = F.gelu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        return x

model = SimpleCNN(width=width).to(device)
print(f"Using SimpleCNN model (BatchNorm + Internal Grid). Total params: {sum(p.numel() for p in model.parameters())}")

################################################################
# Training Setup
################################################################
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# --- CHANGED: Use LpLoss (Mean reduction for comparable % error) ---
loss_fn = LpLoss(size_average=True, reduction=True)
# ------------------------------------------------------------------

################################################################
# Trainer
################################################################
class VerboseTrainer:
    def __init__(self, model, device, n_epochs, optimizer):
        self.model = model
        self.device = device
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.history = []

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
                # Summing the batch means to get the total average later
                # Since loss_fn returns a mean, we multiply by batch size to sum properly
                batch_loss = loss_fn(out, y).item()
                total_loss += batch_loss * x.size(0)
        self.model.train()
        return total_loss / len(loader.dataset)

    def train(self, train_loader, test_loaders=None, training_loss=None, eval_losses=None, scheduler=None):
        for epoch in range(self.n_epochs):
            start = time.time()
            if scheduler:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']

            total_train_loss = 0
            for idx, batch in enumerate(train_loader):
                # loss_fn returns mean over batch. Multiply by batch size to track total sum.
                batch_loss = self.train_one_batch(idx, batch, training_loss)
                total_train_loss += batch_loss * batch['x'].size(0)

            # Divide by total samples to get the average per-sample LpLoss
            train_loss = total_train_loss / len(train_loader.dataset)

            test_losses = {}
            if test_loaders:
                for name, loader in test_loaders.items():
                    test_losses[name] = self.evaluate(loader, eval_losses["LpLoss"])

            if scheduler:
                scheduler.step()

            end = time.time()
            epoch_time = end - start
            test_loss_value = test_losses.get('test', float('nan')) if test_loaders else float('nan')

            epoch_data = {'epoch': epoch + 1, 'time_s': epoch_time, 'train_loss': train_loss, 'test_loss': test_loss_value, 'learning_rate': current_lr}
            self.history.append(epoch_data)

            # Print format: 0.050000 = 5% error
            print(f"Epoch {epoch+1}/{self.n_epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.6f} | Test Loss: {test_loss_value:.6f}")

trainer = VerboseTrainer(model=model, device=device, n_epochs=epochs, optimizer=optimizer)
trainer.train(
    train_loader=train_loader,
    test_loaders={"test": test_loader},
    training_loss=loss_fn,
    eval_losses={"LpLoss": loss_fn},
    scheduler=scheduler
)

pd.DataFrame(trainer.history).to_excel(EXCEL_FILENAME, index=False)
torch.save(model.state_dict(), FINAL_MODEL_FILENAME)

################################################################
# Final Plot
################################################################
print("Generating final comparison plot...")
final_model = SimpleCNN(width=width).to(device)
final_model.load_state_dict(torch.load(FINAL_MODEL_FILENAME, map_location=device))
final_model.eval()

with torch.no_grad():
    batch = next(iter(test_loader2))
    x, y = batch["x"].to(device), batch["y"].to(device)
    out = final_model(x)

    X = x[0, 0].cpu().numpy()
    Y = x[0, 1].cpu().numpy()
    truth = y[0].squeeze().cpu().numpy()
    pred = out[0].squeeze().cpu().numpy()

    nx, ny = 40 // r1, 20 // r2
    if X.shape[0] > 2 * nx and X.shape[1] > ny:
        X_small, Y_small = X[nx:-nx, :ny], Y[nx:-nx, :ny]
        truth_small, pred_small = truth[nx:-nx, :ny], pred[nx:-nx, :ny]
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
    im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small, vmax=vmax)
    fig.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title('Ground Truth (Zoom)')

    im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small, vmax=vmax)
    fig.colorbar(im, ax=ax[1, 1])
    ax[1, 1].set_title('Prediction (Zoom)')

    im = ax[2, 1].pcolormesh(X_small, Y_small, pred_small - truth_small, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 1])
    ax[2, 1].set_title('Difference (Zoom)')

    plt.tight_layout()
    plt.savefig(FINAL_PLOT_FILENAME)
    print(f"Saved comparison plot to {FINAL_PLOT_FILENAME}")