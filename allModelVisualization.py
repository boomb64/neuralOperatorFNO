import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from neuralop.models import FNO

# Force popup window
matplotlib.use('TkAgg')

# ================= CONFIGURATION =================
BASE_PATH = "."
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")
MODEL_DIR = os.path.join(BASE_PATH, "model")

# File Paths
PATH_CNN = os.path.join(MODEL_DIR, "cnn_model.pth")
PATH_FNO = os.path.join(MODEL_DIR, "naca_final_model_cpu.pth")
PATH_PINN = os.path.join(MODEL_DIR, "pinn_model.pth")

# Data Config
s1, s2 = 221, 51
SAMPLE_INDEX = 1005  # Test set index
ZOOM_X = [-0.5, 1.5]
ZOOM_Y = [-0.8, 0.8]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= MODEL ARCHITECTURES =================

# 1. CNN (Must match training architecture)
class SimpleCNN(nn.Module):
    def __init__(self, width=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, width, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.conv3 = nn.Conv2d(width * 2, width * 4, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.conv4 = nn.Conv2d(width * 4, width * 2, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(width * 2)
        self.conv5 = nn.Conv2d(width * 2, width, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(width)
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


# 2. PINN Loader
def load_pinn_model(path):
    model = FNO(
        n_modes=(12, 12),
        hidden_channels=64,
        in_channels=2,
        out_channels=5,
        n_layers=4,
        non_linearity=torch.nn.GELU()
    ).to(device)

    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device, weights_only=False)
            if list(state.keys())[0].startswith('module.'):
                state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            print(f"[OK] Loaded PINN")
        except:
            print(f"[!] PINN Mismatch")
    else:
        print(f"[Missing] PINN")
    model.eval()
    return model


# 3. Manual FNO Loader
def load_manual_fno(path):
    model = FNO(
        n_modes=(24, 12),
        hidden_channels=32,
        in_channels=4,
        out_channels=1,
        n_layers=4,
        domain_padding=None,
        non_linearity=torch.nn.GELU()
    ).to(device)

    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(state, strict=False)
            print(f"[OK] Loaded Manual FNO")
        except Exception as e:
            print(f"[!] FNO Load Error: {e}")
    else:
        print(f"[Missing] Manual FNO")
    model.eval()
    return model


# ================= HELPER FUNCTIONS =================

def make_grid_input(x_geom):
    """Adds grid channels for FNO"""
    batch_size, _, h, w = x_geom.shape
    gridx = torch.linspace(0, 1, h, device=device).reshape(1, 1, h, 1).repeat([batch_size, 1, 1, w])
    gridy = torch.linspace(0, 1, w, device=device).reshape(1, 1, 1, w).repeat([batch_size, 1, h, 1])
    return torch.cat((x_geom, gridx, gridy), dim=1)


# ================= MAIN =================

def main():
    print("--- 1. Loading Data ---")
    try:
        X_all = np.load(os.path.join(DATA_PATH, "NACA_Cylinder_X.npy"))
        Y_all = np.load(os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy"))
        Q_all = np.load(os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy"))

        x_phys = X_all[SAMPLE_INDEX]
        y_phys = Y_all[SAMPLE_INDEX]

        # === FIX: Use Channel 4 (Matches Training) ===
        if Q_all.ndim == 4:
            gt_val = Q_all[SAMPLE_INDEX, 4]
        else:
            gt_val = Q_all[SAMPLE_INDEX]

    except Exception as e:
        print(f"CRITICAL DATA ERROR: {e}")
        return

    print("--- 2. Predicting ---")
    geom_tensor = torch.tensor(np.stack([x_phys, y_phys], axis=0), dtype=torch.float).unsqueeze(0).to(device)

    models_to_plot = []

    # A. Ground Truth
    models_to_plot.append(("Ground Truth (Ch 4)", gt_val))

    # B. CNN
    cnn = SimpleCNN(width=32).to(device)
    if os.path.exists(PATH_CNN):
        try:
            cnn.load_state_dict(torch.load(PATH_CNN, map_location=device, weights_only=False))
            cnn_pred = cnn(geom_tensor).detach().cpu().numpy()[0, 0]
            models_to_plot.append(("CNN", cnn_pred))
            print("[OK] Loaded CNN")
        except Exception as e:
            print(f"[!] CNN Mismatch: {e}")
    else:
        print("[Missing] CNN weights")

    # C. Manual FNO
    fno_input = make_grid_input(geom_tensor)
    fno = load_manual_fno(PATH_FNO)
    with torch.no_grad():
        fno_pred = fno(fno_input).cpu().numpy()[0, 0]
        models_to_plot.append(("Manual FNO", fno_pred))

    # D. PINN
    pinn = load_pinn_model(PATH_PINN)
    with torch.no_grad():
        pinn_out = pinn(geom_tensor).cpu().numpy()[0]
        # === FIX: Extract Channel 4 from PINN output ===
        if pinn_out.shape[0] > 4:
            pinn_pred = pinn_out[4]
        else:
            pinn_pred = pinn_out[0]
        models_to_plot.append(("PINN (Ch 4)", pinn_pred))

    print("--- 3. Visualizing ---")
    num_cols = len(models_to_plot)
    fig, axes = plt.subplots(2, num_cols, figsize=(4 * num_cols, 8))

    # Calculate limits based on Ground Truth Channel 4
    vmin, vmax = gt_val.min(), gt_val.max()

    for i, (name, pred) in enumerate(models_to_plot):
        # Row 1: Prediction
        ax = axes[0, i]
        cf = ax.pcolormesh(x_phys, y_phys, pred, shading='gouraud', cmap='turbo', vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(ZOOM_X)
        ax.set_ylim(ZOOM_Y)
        ax.axis('off')

        # Row 2: Error
        ax_err = axes[1, i]
        if i == 0:
            ax_err.text(0.5, 0.5, "Reference", ha='center')
            ax_err.axis('off')
        else:
            diff = np.abs(gt_val - pred)
            l2 = np.linalg.norm(diff) / np.linalg.norm(gt_val)
            ax_err.pcolormesh(x_phys, y_phys, diff, shading='gouraud', cmap='inferno', vmin=0, vmax=(vmax - vmin) * 0.2)
            ax_err.set_title(f"Rel L2: {l2:.4f}", fontsize=12, color='red')
            ax_err.set_aspect('equal')
            ax_err.set_xlim(ZOOM_X)
            ax_err.set_ylim(ZOOM_Y)
            ax_err.axis('off')

    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Target Field Value')
    plt.suptitle(f"Model Comparison (Test Sample {SAMPLE_INDEX})", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()