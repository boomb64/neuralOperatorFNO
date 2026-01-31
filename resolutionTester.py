"""
OLD FILE: This is an old version of resolution testing. The current version is resolutionFNOCorrect.py
correctFNOresolutionRUNNER.py runs it. (I know my naming convention needs a lot of work)


MULTI-RESOLUTION ROBUSTNESS TESTER
----------------------------------
Function:
    Systematically evaluates the FNO model's ability to maintain accuracy when trained on progressively
    coarser (subsampled) grids, proving its "Resolution Invariance."

Key Features:
    1. Dynamic Subsampling: Accepts an argument `--subsample_r` (e.g., 1, 2, 4, 8) to train on
       1/r resolution data (simulating sparse sensor data).
    2. Zero-Shot Upscaling: After training on coarse data, it evaluates the model on the *Full Resolution*
       test set to measure how well the operator learned the continuous physics.
    3. Automated Logging: Appends results (Grid Size vs. Error) to a summary CSV, allowing for easy
       creation of "Error vs. Resolution" Pareto curves.
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

# We will use the custom LpLoss class, so no import from neuralop.losses

# --- Make reproducible ---
torch.manual_seed(0)
np.random.seed(0)
# Comment out CUDA specific seeds if running on CPU
# torch.cuda.manual_seed(0)
# torch.backends.cudnn.deterministic = True

# --- Argument Parser ---
parser = argparse.ArgumentParser(description='FNO NACA Airfoil Training with Subsampling')
parser.add_argument('--subsample_r', type=int, default=1,
                    help='Subsampling rate (r). r=1: full res (221x51), r=2: half res (~110x25), etc.')
args = parser.parse_args()

################################################################
# Configs
################################################################
r = args.subsample_r
print(f"--- Running with Subsample Rate (r) = {r} ---")

BASE_PATH = "."  # Assume data/ and results/ are in the same dir as the script
DATA_PATH = os.path.join(BASE_PATH, 'data', 'naca')
MODEL_SAVE_DIR = os.path.join(BASE_PATH, 'model_checkpoints_naca')
PLOTS_SAVE_DIR = os.path.join(BASE_PATH, 'results', 'naca_plots_neuralop')

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

FINAL_MODEL_FILENAME = os.path.join(MODEL_SAVE_DIR, f'naca_fno_final_r{r}.pth')
FINAL_PLOT_FILENAME = os.path.join(PLOTS_SAVE_DIR, f'naca_final_comparison_r{r}.png')
SUMMARY_FILENAME = os.path.join(PLOTS_SAVE_DIR, 'naca_experiment_summary.csv')
EXCEL_LOG_FILENAME = os.path.join(PLOTS_SAVE_DIR, f'naca_training_log_r{r}.xlsx')  # --- MODIFIED ---

INPUT_X_PATH = os.path.join(DATA_PATH, 'NACA_Cylinder_X.npy')
INPUT_Y_PATH = os.path.join(DATA_PATH, 'NACA_Cylinder_Y.npy')
OUTPUT_Q_PATH = os.path.join(DATA_PATH, 'NACA_Cylinder_Q.npy')

# Training params (match naca_geofno.py)
ntrain = 1000
ntest = 200
batch_size = 20
learning_rate = 0.001
epochs = 501
step_size = 100
gamma = 0.5
weight_decay = 1e-4

# Model params (match naca_geofno.py)
base_modes = 12
modes1 = max((base_modes * 2) // r, 1)  # Scale modes based on r
modes2 = max(base_modes // r, 1)  # Scale modes based on r
width = 32
print(f"Using Fourier modes: ({modes1}, {modes2})")

# Data params (match naca_geofno.py)
r1 = r
r2 = r
s1_full = 221  # Full grid size x
s2_full = 51  # Full grid size y
# Calculate new subsampled grid size
s1 = int(((s1_full - 1) / r1) + 1)
s2 = int(((s2_full - 1) / r2) + 1)
print(f"Original grid: ({s1_full}, {s2_full}). Subsampled training grid: ({s1}, {s2})")

# Device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


################################################################
# Helper Functions
################################################################
def get_normalized_grid(batchsize, size_x, size_y, device):
    """
    Generates a normalized (0-1) grid and repeats it for the batch.
    """
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def count_params(model):
    """
    Counts the number of trainable parameters in a model.
    """
    c = 0
    try:
        params = list(model.parameters())
        if not params:
            print("Warning: Model has no parameters.")
            return 0
        for p in params:
            if p.requires_grad:  # Only count trainable parameters
                c += reduce(operator.mul, list(p.size()))
    except Exception as e:
        print(f"Error accessing model parameters: {e}")
        return -1
    return c


# --- Copied LpLoss class from old utilities3.py ---
# This ensures we are using the *exact* relative loss from the paper
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        # Assume uniform mesh
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
        # Add epsilon to denominator to prevent division by zero
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


# --- End of LpLoss class ---

# --- Dataset class returning dicts (for compatibility with newer Trainers if needed) ---
class DictDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Returns a dictionary, as expected by the new Trainer
        return {"x": self.x[idx], "y": self.y[idx]}


################################################################
# Load data and prepare for FNO
################################################################
inputX = np.load(INPUT_X_PATH)
inputX = torch.tensor(inputX, dtype=torch.float)  # Shape: [N, 221, 51]
inputY = np.load(INPUT_Y_PATH)
inputY = torch.tensor(inputY, dtype=torch.float)  # Shape: [N, 221, 51]

input_phys_coords = torch.stack([inputX, inputY], dim=-1)  # Shape: [N, 221, 51, 2]

output_all = np.load(OUTPUT_Q_PATH)  # Shape: [N, 5, 221, 51]
if output_all.ndim == 4 and output_all.shape[1] >= 5:
    output = torch.tensor(output_all[:, 4], dtype=torch.float)  # Shape: [N, 221, 51]
else:
    raise ValueError(f"Unexpected shape for OUTPUT_Q_PATH: {output_all.shape}. Expected (N, 5, {s1_full}, {s2_full})")

print(f"Loaded full-res input coords shape: {input_phys_coords.shape}, Output shape: {output.shape}")

# --- 1. Create (Full-Res) Data for FINAL TEST ---
# We need this for the r=1 test and the final "zero-shot" evaluation
grid_normalized_full = get_normalized_grid(input_phys_coords.shape[0], s1_full, s2_full,
                                           device='cpu')  # Shape: [N, 221, 51, 2]
input_features_full = torch.cat((input_phys_coords, grid_normalized_full), dim=-1)  # Shape: [N, 221, 51, 4]
input_features_full = input_features_full.permute(0, 3, 1, 2)  # Shape: [N, 4, 221, 51]
output_full = output.unsqueeze(1)  # Shape: [N, 1, 221, 51]

x_test_full = input_features_full[ntrain:ntrain + ntest]
y_test_full = output_full[ntrain:ntrain + ntest]

print(f"Final x_test (full) shape: {x_test_full.shape}, y_test (full) shape: {y_test_full.shape}")

full_res_test_dataset = DictDataset(x_test_full, y_test_full)
full_res_loader = DataLoader(full_res_test_dataset, batch_size=batch_size, shuffle=False)
plot_loader = DataLoader(full_res_test_dataset, batch_size=1, shuffle=False)

# --- 2. Create Subsampled Data for TRAINING ---
# Slicing the data using the resolution parameter r
input_phys_coords_sub = input_phys_coords[:, ::r1, ::r2][:, :s1, :s2, :]  # Shape: [N, s1, s2, 2]
output_sub = output[:, ::r1, ::r2][:, :s1, :s2]  # Shape: [N, s1, s2]

# Generate normalized grid for the new subsampled shape
grid_normalized = get_normalized_grid(input_phys_coords_sub.shape[0], s1, s2, device='cpu')  # Shape: [N, s1, s2, 2]

# Concatenate subsampled physical coords and new grid
input_features = torch.cat((input_phys_coords_sub, grid_normalized), dim=-1)  # Shape: [N, s1, s2, 4]

# Permute for FNO: (Batch, Channels, X, Y)
input_features = input_features.permute(0, 3, 1, 2)  # Shape: [N, 4, s1, s2]

# Add channel dimension to output
output_sub = output_sub.unsqueeze(1)  # Shape: [N, 1, s1, s2]

# Split subsampled data
x_train = input_features[:ntrain]
y_train = output_sub[:ntrain]
x_test_sub = input_features[ntrain:ntrain + ntest]
y_test_sub = output_sub[ntrain:ntrain + ntest]

print(f"Final x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"Final x_test (sub) shape: {x_test_sub.shape}, y_test (sub) shape: {y_test_sub.shape}")

# Create Datasets and DataLoaders for subsampled data
train_dataset = DictDataset(x_train, y_train)
sub_test_dataset = DictDataset(x_test_sub, y_test_sub)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
sub_test_loader = DataLoader(sub_test_dataset, batch_size=batch_size, shuffle=False)  # Loader for low-res testing

################################################################
# Build the model using neuralop.models.FNO
################################################################
model = FNO(
    n_modes=(modes1, modes2),
    hidden_channels=width,
    in_channels=4,  # 2 physical coords + 2 normalized grid coords
    out_channels=1,  # Predicting 1 variable
).to(device)

print(f"NeuralOperator FNO parameter count: {count_params(model)}")

################################################################
# Set up training components
################################################################
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Use the custom LpLoss class we pasted above
print("Using custom-defined LpLoss (relative=True).")
training_loss_fn = LpLoss(size_average=True)
eval_loss_fn = LpLoss(size_average=True)

################################################################
# Training loop (Manual)
################################################################
print(f"Starting training for r={r}...")

training_history = []  # --- MODIFIED ---
overall_start_time = default_timer()
final_train_loss = 0.0
final_test_loss_sub = 0.0
final_test_loss_full = 0.0

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    current_lr = optimizer.param_groups[0]['lr']  # --- MODIFIED: Get LR at start of epoch ---

    for batch in train_loader:
        # Data comes as a dict from our custom Dataset
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        optimizer.zero_grad()
        out = model(x)

        # Flatten spatial dims for LpLoss
        loss = training_loss_fn(out.view(x.size(0), -1), y.view(y.size(0), -1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2_sub = 0.0
    with torch.no_grad():
        for batch in sub_test_loader:  # Test on subsampled test data
            x = batch['x'].to(device)
            y = batch['y'].to(device)

            out = model(x)
            test_l2_sub += eval_loss_fn(out.view(x.size(0), -1), y.view(y.size(0), -1)).item()

    train_l2 /= len(train_loader)
    test_l2_sub /= len(sub_test_loader)

    t2 = default_timer()
    epoch_time = t2 - t1  # --- MODIFIED ---

    # --- MODIFIED: Log data to history ---
    epoch_data = {
        'epoch': ep + 1,
        'time_s': epoch_time,
        'train_loss': train_l2,
        'test_loss_sub': test_l2_sub,
        'learning_rate': current_lr
    }
    training_history.append(epoch_data)
    # --- END MODIFICATION ---

    # --- MODIFIED: Updated print to be 1-based and use epoch_time variable ---
    print(
        f"Epoch {ep + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_l2:.6f} | Test Loss (Sub, r={r}): {test_l2_sub:.6f}")

    # Store the final epoch's losses
    if ep == epochs - 1:
        final_train_loss = train_l2
        final_test_loss_sub = test_l2_sub

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

# --- Run Final Evaluation on FULL-RESOLUTION Test Set ---
print(f"\nRunning final evaluation on full-resolution (r=1) test set...")
model.eval()
test_l2_full = 0.0
with torch.no_grad():
    for batch in full_res_loader:  # Use the full-res loader
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        # Pass the full-res input; FNO is resolution-invariant
        out = model(x)

        # Calculate loss against full-res truth
        test_l2_full += eval_loss_fn(out.view(x.size(0), -1), y.view(y.size(0), -1)).item()

final_test_loss_full = test_l2_full / len(full_res_loader)
print(f"--- Final Test Loss (Full Res, r=1): {final_test_loss_full:.6f} ---")

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
        if summary_file_exists:  # Write header only if file is new
            writer.writerow(
                ['r', 'Grid_Size', 'Modes', 'Final_Train_Loss(r)', 'Final_Test_Loss(r)', 'Final_Test_Loss(r=1)',
                 'Total_Time_sec'])

        grid_size_str = f"{s1}x{s2}"
        modes_str = f"({modes1}, {modes2})"
        writer.writerow([r, grid_size_str, modes_str, f"{final_train_loss:.6f}", f"{final_test_loss_sub:.6f}",
                         f"{final_test_loss_full:.6f}", f"{overall_time_taken:.2f}"])
    print(f"Appended results to {SUMMARY_FILENAME}")
except Exception as e:
    print(f"Error saving summary to CSV: {e}")

print("Generating final comparison plot...")
model.eval()

with torch.no_grad():
    # Get a full-res sample for plotting
    batch = next(iter(plot_loader))
    x_plot_features = batch['x'].to(device)
    y_plot_truth = batch['y'].to(device)

    # Get the prediction (on full-res)
    out_plot = model(x_plot_features)

    # --- Prepare data for plotting (all full-res) ---
    # Need to load the full-res (r=1) physical coordinates for the plot
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
        # Fallback if slicing indices are out of bounds
        X_small, Y_small, truth_small, pred_small = x_coords_plot, y_coords_plot, truth, pred

    # --- Create the plot ---
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(16, 16))
    vmin, vmax = np.min(truth), np.max(truth)

    im = ax[0, 0].pcolormesh(x_coords_plot, y_coords_plot, truth, shading='gouraud', cmap='viridis', vmin=vmin,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title(f'Ground Truth (Full) r=1')

    im = ax[1, 0].pcolormesh(x_coords_plot, y_coords_plot, pred, shading='gouraud', cmap='viridis', vmin=vmin,
                             vmax=vmax)
    fig.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_title(f'Prediction (from r={r} model)')

    im = ax[2, 0].pcolormesh(x_coords_plot, y_coords_plot, pred - truth, shading='gouraud', cmap='coolwarm')
    fig.colorbar(im, ax=ax[2, 0])
    ax[2, 0].set_title(f'Difference (from r={r} model)')

    # Check if small arrays are valid before plotting
    if X_small.size > 0 and Y_small.size > 0 and truth_small.size > 0 and pred_small.size > 0:
        vmin_small, vmax_small = np.min(truth_small), np.max(truth_small)
        im = ax[0, 1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                                 vmax=vmax_small)
        fig.colorbar(im, ax=ax[0, 1])
        ax[0, 1].set_title(f'Ground Truth (Zoom) r=1')

        im = ax[1, 1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', cmap='viridis', vmin=vmin_small,
                                 vmax=vmax_small)
        fig.colorbar(im, ax=ax[1, 1])
        ax[1, 1].set_title(f'Prediction (Zoom) r={r}')

        im = ax[2, 1].pcolormesh(X_small, Y_small, np.abs(pred_small - truth_small), shading='gouraud', cmap='magma')
        fig.colorbar(im, ax=ax[2, 1])
        ax[2, 1].set_title(f'Absolute Error (Zoom) r={r}')
    else:
        print(f"Skipping zoom plots for r={r} as resolution is too low for slicing.")
        ax[0, 1].set_title('Zoom plot skipped (resolution too low)')
        ax[1, 1].set_title('Zoom plot skipped (resolution too low)')
        ax[2, 1].set_title('Zoom plot skipped (resolution too low)')

    plt.tight_layout()
    fig.savefig(FINAL_PLOT_FILENAME)
    plt.close(fig)
    print(f"Saved final comparison plot to {FINAL_PLOT_FILENAME}")