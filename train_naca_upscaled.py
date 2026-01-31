"""
UPSCALING TRAINING WORKER (Synthetic Super-Resolution)
------------------------------------------------------
Function:
    The "Worker" script that trains an FNO model on artificially upscaled data.
    It is designed to be called by 'runUpscaleExperiments.py' with different scale factors.

Key Features:
    1. Synthetic Density: Uses bilinear interpolation to increase the grid size of the training data
       (e.g., s=2.0 turns 221x51 into 442x102) before the model sees it.
    2. Adaptive Capacity: Automatically scales the number of Fourier Modes to match the increased
       resolution, allowing the model to capture higher-frequency details if they exist.
    3. Backward Validation: Evaluates the trained high-res model on the *original* low-res test set
       to check if learning on dense synthetic grids improves real-world accuracy.
"""

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import operator
from functools import reduce
from timeit import default_timer
import argparse
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pandas as pd  # --- MODIFIED ---

# --- Use neuraloperator library components ---
from neuralop.models import FNO

# --- Make reproducible ---
torch.manual_seed(0)
np.random.seed(0)
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='FNO NACA Airfoil Upscaling Experiment')
parser.add_argument('--scale_factor', type=float, default=1.0,
                    help='Upscaling factor (s). s=1.0: baseline, s=2.0: train on 442x102')
args = parser.parse_args()

################################################################
# Configs
################################################################
scale = args.scale_factor
print(f"--- Running with Upscale Factor (s) = {scale} ---")

BASE_PATH = "."
DATA_PATH = os.path.join(BASE_PATH, 'data', 'naca')
MODEL_SAVE_DIR = os.path.join(BASE_PATH, 'model_checkpoints_naca_upscale')
PLOTS_SAVE_DIR = os.path.join(BASE_PATH, 'results', 'naca_plots_neuralop_upscale')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

FINAL_MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, f'naca_fno_final_s{scale}.pth')
FINAL_PLOT_FILENAME = os.path.join(PLOTS_SAVE_DIR, f'naca_final_comparison_s{scale}.png')
SUMMARY_FILENAME = os.path.join(PLOTS_SAVE_DIR, 'naca_upscale_summary.csv')  # Summary for this experiment
EXCEL_LOG_FILENAME = os.path.join(PLOTS_SAVE_DIR, f'naca_upscale_log_s{scale}.xlsx')  # --- MODIFIED ---

INPUT_X_PATH = os.path.join(DATA_PATH, 'NACA_Cylinder_X.npy')
INPUT_Y_PATH = os.path.join(DATA_PATH, 'NACA_Cylinder_Y.npy')
OUTPUT_Q_PATH = os.path.join(DATA_PATH, 'NACA_Cylinder_Q.npy')

# Training params
ntrain = 1000
ntest = 200
batch_size = 20
learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5
weight_decay = 1e-4

# Model params
base_modes = 12
# --- MODIFICATION: Scale modes UP ---
modes1 = max(int((base_modes * 2) * scale), 1)
modes2 = max(int(base_modes * scale), 1)
width = 32
print(f"Using Fourier modes: ({modes1}, {modes2})")

# Data params
s1_full = 221
s2_full = 51
# --- MODIFICATION: Calculate NEW upscaled grid size ---
s1_new = int(s1_full * scale)
s2_new = int(s2_full * scale)
# --- END MODIFICATION ---
print(f"Original grid: ({s1_full}, {s2_full}). New upscaled training grid: ({s1_new}, {s2_new})")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


################################################################
# Helper Functions
################################################################
def get_normalized_grid(batchsize, size_x, size_y, device):
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def count_params(model):
    c = 0
    try:
        params = list(model.parameters())
        if not params: return 0
        for p in params:
            if p.requires_grad: c += reduce(operator.mul, list(p.size()))
    except Exception as e:
        print(f"Error counting params: {e}")
        return -1
    return c


# Pasted LpLoss class from utilities3.py
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        h = 1.0 / (x.size()[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
                                                          self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        # Add epsilon to prevent division by zero
        y_norms = torch.where(y_norms == 0, torch.tensor(1e-6, device=y.device), y_norms)
        relative_error = diff_norms / y_norms
        if self.reduction:
            if self.size_average:
                return torch.mean(relative_error)
            else:
                return torch.sum(relative_error)
        return relative_error

    def __call__(self, x, y):
        return self.rel(x, y)


# Dataset class returning dicts
class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {"x": self.x[idx], "y": self.y[idx]}


################################################################
# Load data and prepare for FNO
################################################################
inputX = np.load(INPUT_X_PATH)
inputX = torch.tensor(inputX, dtype=torch.float)  # Shape: [N, 221, 51]
inputY = np.load(INPUT_Y_PATH)
inputY = torch.tensor(inputY, dtype=torch.float)  # Shape: [N, 221, 51]

input_phys_coords = torch.stack([inputX, inputY], dim=-1)  # Shape: [N, 221, 51, 2]

output_all = np.load(OUTPUT_Q_PATH)
if output_all.ndim == 4 and output_all.shape[1] >= 5:
    output = torch.tensor(output_all[:, 4], dtype=torch.float)  # Shape: [N, 221, 51]
else:
    raise ValueError(f"Unexpected shape for OUTPUT_Q_PATH: {output_all.shape}. Expected (N, 5, {s1_full}, {s2_full})")

print(f"Loaded full-res input coords shape: {input_phys_coords.shape}, Output shape: {output.shape}")

# --- 1. Create (Full-Res) Data ---
grid_normalized_full = get_normalized_grid(input_phys_coords.shape[0], s1_full, s2_full,
                                           device='cpu')  # Shape: [N, 221, 51, 2]
input_features_full = torch.cat((input_phys_coords, grid_normalized_full), dim=-1)  # Shape: [N, 221, 51, 4]
input_features_full = input_features_full.permute(0, 3, 1, 2)  # Shape: [N, 4, 221, 51]
output_full = output.unsqueeze(1)  # Shape: [N, 1, 221, 51]

# --- 2. Create (Full-Res) Test Set (for final evaluation) ---
x_test_orig = input_features_full[ntrain:ntrain + ntest]
y_test_orig = output_full[ntrain:ntrain + ntest]
print(f"Final x_test (original) shape: {x_test_orig.shape}, y_test (original) shape: {y_test_orig.shape}")
orig_res_test_dataset = DictDataset(x_test_orig, y_test_orig)
orig_res_loader = DataLoader(orig_res_test_dataset, batch_size=batch_size, shuffle=False)
plot_loader = DataLoader(orig_res_test_dataset, batch_size=1, shuffle=False)

# --- 3. Create Upscaled Data for TRAINING ---
print(f"Upscaling training data to {s1_new}x{s2_new}...")
# Get only the training portion of the full-res data
x_train_orig = input_features_full[:ntrain]
y_train_orig = output_full[:ntrain]

# Use interpolate to upscale
x_train_upscaled = F.interpolate(x_train_orig, size=(s1_new, s2_new), mode='bilinear', align_corners=True)
y_train_upscaled = F.interpolate(y_train_orig, size=(s1_new, s2_new), mode='bilinear', align_corners=True)
print("Upscaling complete.")

# --- 4. Create Upscaled Data for epoch-wise TESTING ---
x_test_upscaled = F.interpolate(x_test_orig, size=(s1_new, s2_new), mode='bilinear', align_corners=True)
y_test_upscaled = F.interpolate(y_test_orig, size=(s1_new, s2_new), mode='bilinear', align_corners=True)

print(f"Final x_train (upscaled) shape: {x_train_upscaled.shape}, y_train (upscaled) shape: {y_train_upscaled.shape}")
print(f"Final x_test (upscaled) shape: {x_test_upscaled.shape}, y_test (upscaled) shape: {y_test_upscaled.shape}")

# Create DictDatasets and DataLoaders for upscaled data
train_dataset = DictDataset(x_train_upscaled, y_train_upscaled)
upscaled_test_dataset = DictDataset(x_test_upscaled, y_test_upscaled)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
upscaled_test_loader = DataLoader(upscaled_test_dataset, batch_size=batch_size, shuffle=False)

################################################################
# Build the model using neuralop.models.FNO
################################################################
model = FNO(
    n_modes=(modes1, modes2),  # Use the NEW, larger mode counts
    hidden_channels=width,
    in_channels=4,
    out_channels=1,
).to(device)

print(f"NeuralOperator FNO parameter count: {count_params(model)}")

################################################################
# Set up training components
################################################################
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

print("Using custom-defined LpLoss (relative=True).")
training_loss_fn = LpLoss(size_average=True)
eval_loss_fn = LpLoss(size_average=True)

################################################################
# Training loop (Manual)
################################################################
print(f"Starting training on upscaled (s={scale}) data...")

training_history = []  # --- MODIFIED ---
overall_start_time = default_timer()
final_train_loss = 0.0
final_test_loss_scaled = 0.0
final_test_loss_orig = 0.0

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    current_lr = optimizer.param_groups[0]['lr']  # --- MODIFIED: Get LR ---

    for batch in train_loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        out = model(x)

        loss = training_loss_fn(out.view(x.size(0), -1), y.view(y.size(0), -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2_scaled = 0.0
    with torch.no_grad():
        for batch in upscaled_test_loader:  # Test on upscaled test data
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            out = model(x)
            test_l2_scaled += eval_loss_fn(out.view(x.size(0), -1), y.view(y.size(0), -1)).item()

    train_l2 /= len(train_loader)
    test_l2_scaled /= len(upscaled_test_loader)

    t2 = default_timer()
    epoch_time = t2 - t1  # --- MODIFIED ---

    # --- MODIFIED: Log data to history ---
    epoch_data = {
        'epoch': ep + 1,
        'time_s': epoch_time,
        'train_loss': train_l2,
        'test_loss_scaled': test_l2_scaled,
        'learning_rate': current_lr
    }
    training_history.append(epoch_data)
    # --- END MODIFICATION ---

    # --- MODIFIED: Updated print to be 1-based and use epoch_time ---
    print(
        f"Epoch {ep + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_l2:.6f} | Test Loss (Scaled, s={scale}): {test_l2_scaled:.6f}")

    if ep == epochs - 1:
        final_train_loss = train_l2
        final_test_loss_scaled = test_l2_scaled

print("Training finished.")
overall_time_taken = default_timer() - overall_start_time
print(f"Total training time: {overall_time_taken:.2f}s")

# --- MODIFIED: Save history to Excel ---
print(f"Saving training history to {EXCEL_LOG_FILENAME}...")
try:
    df_history = pd.DataFrame(training_history)
    df_history.to_excel(EXCEL_LOG_FILENAME, index=False)
    print("History saved successfully.")
except Exception as e:
    print(f"Error saving history to Excel: {e}")
# --- END MODIFICATION ---

# --- Run Final Evaluation on ORIGINAL-RESOLUTION Test Set ---
print(f"\nRunning final evaluation on original-resolution (s=1) test set...")
model.eval()
test_l2_orig = 0.0
with torch.no_grad():
    for batch in orig_res_loader:  # Use the original resolution loader
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        # We can pass the original-res input; FNO is resolution-invariant
        out = model(x)

        test_l2_orig += eval_loss_fn(out.view(x.size(0), -1), y.view(y.size(0), -1)).item()

final_test_loss_orig = test_l2_orig / len(orig_res_loader)
print(f"--- Final Test Loss (Original Res, s=1): {final_test_loss_orig:.6f} ---")

################################################################
# Save final model and generate plot
################################################################
torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
print(f"Saved final model state_dict to {FINAL_MODEL_FILENAME}")

# --- Save results to summary CSV ---
summary_file_exists = not os.path.exists(SUMMARY_FILENAME)  # Check if file *doesn't* exist
try:
    with open(SUMMARY_FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        if summary_file_exists:
            writer.writerow(
                ['s', 'Train_Grid', 'Modes', 'Final_Train_Loss(s)', 'Final_Test_Loss(s)', 'Final_Test_Loss(s=1)',
                 'Total_Time_sec'])

        grid_size_str = f"{s1_new}x{s2_new}"
        modes_str = f"({modes1}, {modes2})"
        writer.writerow([scale, grid_size_str, modes_str, f"{final_train_loss:.6f}", f"{final_test_loss_scaled:.6f}",
                         f"{final_test_loss_orig:.6f}", f"{overall_time_taken:.2f}"])
    print(f"Appended results to {SUMMARY_FILENAME}")
except Exception as e:
    print(f"Error saving summary to CSV: {e}")

print("Generating final comparison plot...")
model.eval()

with torch.no_grad():
    batch = next(iter(plot_loader))  # Get an original-res sample
    x_plot_features = batch['x'].to(device)
    y_plot_truth = batch['y'].to(device)

    out_plot = model(x_plot_features)  # Predict on original-res

    # --- Prepare data for plotting (all original-res) ---
    x_coords_plot = np.load(INPUT_X_PATH)[ntrain:ntrain + 1, ::1, ::1][:, :s1_full, :s2_full].squeeze()
    y_coords_plot = np.load(INPUT_Y_PATH)[ntrain:ntrain + 1, ::1, ::1][:, :s1_full, :s2_full].squeeze()

    truth = y_plot_truth[0, 0].squeeze().detach().cpu().numpy()
    pred = out_plot[0, 0].squeeze().detach().cpu().numpy()

    # Slicing for zoomed view (use r=1 for slicing)
    nx = 40 // 1
    ny = 20 // 1
    if x_coords_plot.shape[0] > 2 * nx and x_coords_plot.shape[1] > ny and nx > 0 and ny > 0:
        X_small = x_coords_plot[nx:-nx, :ny]
        Y_small = y_coords_plot[nx:-nx, :ny]
        truth_small = truth[nx:-nx, :ny]
        pred_small = pred[nx:-nx, :ny]
    else:
        X_small, Y_small, truth_small, pred_small = x_coords_plot, y_coords_plot, truth, pred

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    im = ax[0, 0].pcolormesh(x_coords_plot, y_coords_plot, truth, shading='gouraud', cmap='viridis', vmin=vmin,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title(f'Ground Truth (Original Res)')

    im = ax[1, 0].pcolormesh(x_coords_plot, y_coords_plot, pred, shading='gouraud', cmap='viridis', vmin=vmin,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title(f'Prediction (from s={scale} model)')

    im = ax[2, 0].pcolormesh(x_coords_plot, y_coords_plot, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title(f'Difference (from s={scale} model)')

    if X_small.size > 0 and Y_small.size > 0 and truth_small.size > 0 and pred_small.size > 0:
        vmin_small, vmax_small = np.min(truth_small), np.max(truth_small)
        im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                                 vmax=vmax_small)
        fig.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_title(f'Ground Truth (Zoom)')

        im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                                 vmax=vmax_small)
        fig.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_title(f'Prediction (Zoom) (from s={scale} model)')

        im = ax[2, 1].pcolormesh(X_small, Y_small, np.abs(pred_small - truth_small), shading='gouraud', cmap='magma')
        fig.colorbar(im, ax=ax[2, 1])
        ax[2, 1].set_title(f'Absolute Error (Zoom) (from s={scale} model)')
    else:
        print(f"Skipping zoom plots for s={scale} as resolution is too low for slicing.")
        ax[0, 1].set_title('Zoom plot skipped')
        ax[1, 1].set_title('Zoom plot skipped')
        ax[2, 1].set_title('Zoom plot skipped')

    plt.tight_layout()
    fig.savefig(FINAL_PLOT_FILENAME)
    plt.close(fig)
    print(f"Saved final comparison plot to {FINAL_PLOT_FILENAME}")