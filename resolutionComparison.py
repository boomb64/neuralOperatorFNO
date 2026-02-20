"""
FNO ZERO-SHOT BENCHMARKER
-------------------------
Function:
    Loads models trained at various resolutions (r=1, 2, 4, 8) and evaluates
    them all on the SAME 50 Full-Resolution test scenarios.
"""

import os
import torch
import math
import numpy as np
from timeit import default_timer
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from neuralop.models import FNO
from neuralop.losses import LpLoss

# --- Configuration ---
device = torch.device("cuda")
BASE_PATH = "."
MODEL_PATH = os.path.join(BASE_PATH, "model")
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")

INPUT_X = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Sigma = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

# We want 50 test scenarios
ntrain = 1000  # Skip the training data
ntest = 50
width = 32
base_modes = 12

resolutions_to_test = [1, 2, 4, 8]

# --- 1. Load FULL Resolution Data ---
print("Loading Full Resolution Data...")
inputX_full = torch.tensor(np.load(INPUT_X), dtype=torch.float)
inputY_full = torch.tensor(np.load(INPUT_Y), dtype=torch.float)
input_data_full = torch.stack([inputX_full, inputY_full], dim=-1)

output_data_full_np = np.load(OUTPUT_Sigma)
if output_data_full_np.ndim == 4 and output_data_full_np.shape[1] > 4:
    output_data_full = torch.tensor(output_data_full_np[:, 4], dtype=torch.float)
else:
    output_data_full = torch.tensor(output_data_full_np, dtype=torch.float)

# Extract the 50 test samples
x_test_full = input_data_full[ntrain:ntrain + ntest].permute(0, 3, 1, 2).contiguous()
y_test_full = output_data_full[ntrain:ntrain + ntest].unsqueeze(-1).permute(0, 3, 1, 2).contiguous()


class PaddedDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        # Use the SAME dynamic logic as resolutionFNOCorrect.py
        h, w = x.shape[2], x.shape[3]
        self.target_h = math.ceil(h / 8) * 8
        self.target_w = math.ceil(w / 8) * 8

        if self.target_h == h: self.target_h += 8
        if self.target_w == w: self.target_w += 8

        self.pad_h = self.target_h - h
        self.pad_w = self.target_w - w

        gridx = torch.tensor(np.linspace(0, 1, self.target_h), dtype=torch.float).reshape(1, self.target_h, 1).repeat([1, 1, self.target_w])
        gridy = torch.tensor(np.linspace(0, 1, self.target_w), dtype=torch.float).reshape(1, 1, self.target_w).repeat([1, self.target_h, 1])
        self.grid = torch.cat((gridx, gridy), dim=0)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_padded = F.pad(self.x[idx], (0, self.pad_w, 0, self.pad_h))
        y_padded = F.pad(self.y[idx], (0, self.pad_w, 0, self.pad_h))
        x_combined = torch.cat((x_padded, self.grid), dim=0)
        return {"x": x_combined, "y": y_padded}


test_full_ds = PaddedDataset(x_test_full, y_test_full)
# Batch size 1 to measure pure sequential inference time if desired, or test as a batch
test_loader = DataLoader(test_full_ds, batch_size=50, shuffle=False)

loss_fn = LpLoss(d=2, p=2, reduction='sum')

# --- 3. Evaluation Loop ---
results = []

print(f"\nEvaluating {ntest} Full-Resolution Scenarios:")
print("-" * 60)

with torch.no_grad():
    # Get the single batch of 50
    batch = next(iter(test_loader))
    x = batch["x"].to(device)
    y = batch["y"].to(device)

    for r in resolutions_to_test:
        # --- Handle the specific r=1 filename ---
        if r == 1:
            model_filename = os.path.join(MODEL_PATH, "naca_fno_r1.pth")
        else:
            model_filename = os.path.join(MODEL_PATH, f"naca_fno_r{r}.pth")

        if not os.path.exists(model_filename):
            print(f"Skipping r={r} (Model not found at {model_filename})")
            continue

        # Recalculate the exact modes this specific model was trained with
        s1_full, s2_full = 221, 51
        s1 = int(((s1_full - 1) / r) + 1)
        s2 = int(((s2_full - 1) / r) + 1)
        modes_x = min(base_modes * 2, s1 // 2)
        modes_y = min(base_modes, s2 // 2)

        # Initialize model with the correct truncated modes
        model = FNO(
            n_modes=(modes_x, modes_y),
            hidden_channels=width,
            in_channels=4,
            out_channels=1,
            n_layers=4,
            domain_padding=None,
            non_linearity=torch.nn.GELU()
        ).to(device)

        model.load_state_dict(torch.load(model_filename, map_location=device, weights_only=False))
        model.eval()

        # Time the prediction
        t_start = default_timer()
        out = model(x)
        t_end = default_timer()

        # --- THE MISSING LINES ---
        total_time = t_end - t_start
        avg_time_per_sample = total_time / ntest
        # -------------------------

        # Slice back down to the TRUE physical domain (221x51) before computing error
        out_unpadded = out[:, :, :221, :51]
        y_unpadded = y[:, :, :221, :51]

        # Calculate Error and "Accuracy" on the actual geometry
        relative_l2_error = loss_fn(out_unpadded, y_unpadded).item() / ntest
        accuracy = 1.0 - relative_l2_error

        print(f"Model r={r: <2} | Modes: ({modes_x: <2}, {modes_y: <2}) | "
              f"Acc: {accuracy:.4f} | Avg Time/Sample: {avg_time_per_sample:.4f}s")

        results.append({
            "Resolution_Trained": r,
            "Modes_X": modes_x,
            "Modes_Y": modes_y,
            "Relative_L2_Error": relative_l2_error,
            "Accuracy": accuracy,
            "Total_Inference_Time_s": total_time,
            "Avg_Time_Per_Sample_s": avg_time_per_sample
        })

# Optional: Save to CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(BASE_PATH, "zero_shot_benchmark_results.csv"), index=False)
print("-" * 60)
print("Saved results to zero_shot_benchmark_results.csv")