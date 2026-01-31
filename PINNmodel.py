"""
FINAL STABILIZED PINN (Channel Fix)
-----------------------------------
Fixes:
1. Increased out_channels to 5 to match dataset Q file.
2. Updated compute_pinn_loss to slice the correct channels for u, v, p.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from neuralop.models import FNO

# ================= CONFIGURATION =================
BASE_PATH = "."
DATA_PATH = os.path.join(BASE_PATH, "data", "naca")
MODEL_PATH = os.path.join(BASE_PATH, "model")
os.makedirs(MODEL_PATH, exist_ok=True)

INPUT_X_FILE = os.path.join(DATA_PATH, "NACA_Cylinder_X.npy")
INPUT_Y_FILE = os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy")
OUTPUT_Q_FILE = os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy")

FINAL_MODEL_FILENAME = os.path.join(MODEL_PATH, "pinn_model.pth")
EXCEL_FILENAME = os.path.join(MODEL_PATH, "pinn_training_log_manual.xlsx")

# Physics Weights
LAMBDA_PHYS = 0.05  # Reduced slightly for initial stability
RHO = 1.0
NU = 0.001

MODES = 12
WIDTH = 64
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= DATA LOADER =================
class GeoFNODataset(Dataset):
    def __init__(self):
        print(f"Loading files from {DATA_PATH}...")
        x_coords = np.load(INPUT_X_FILE)
        y_coords = np.load(INPUT_Y_FILE)
        q_data = np.load(OUTPUT_Q_FILE)  # [Samples, 5, H, W]

        # Grid input: [Samples, 2, H, W]
        coords = np.stack([x_coords, y_coords], axis=1)

        self.x_data = torch.tensor(coords, dtype=torch.float)
        self.y_data = torch.tensor(q_data, dtype=torch.float)
        print(f"Input Shape: {self.x_data.shape} | Output Shape: {self.y_data.shape}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return {"x": self.x_data[idx], "y": self.y_data[idx]}


# ================= PHYSICS LOSS =================
def compute_pinn_loss(coords, y_pred):
    """
    y_pred has 5 channels. We extract u, v, p.
    """
    # Spacing from physical coords
    dx = 1.0 / (y_pred.shape[-1] - 1)
    dy = 1.0 / (y_pred.shape[-2] - 1)

    # Extract u, v, p based on Geo-FNO channel order
    u = y_pred[:, 0:1, :, :]
    v = y_pred[:, 1:2, :, :]
    p = y_pred[:, 2:3, :, :]

    # Gradients
    u_x = torch.gradient(u, dim=3)[0] / dx
    v_y = torch.gradient(v, dim=2)[0] / dy

    # 1. Continuity
    loss_continuity = torch.mean((u_x + v_y) ** 2)

    # 2. X-Momentum
    u_y = torch.gradient(u, dim=2)[0] / dy
    p_x = torch.gradient(p, dim=3)[0] / dx
    x_mom = (u * u_x + v * u_y + (1 / RHO) * p_x)
    loss_momentum = torch.mean(x_mom ** 2)

    return loss_continuity + loss_momentum


# ================= MAIN =================
def main():
    dataset = GeoFNODataset()
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # FIX: Set out_channels=5 to match dataset
    model = FNO(n_modes=(MODES, MODES),
                hidden_channels=WIDTH,
                in_channels=2,
                out_channels=5).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    data_criterion = nn.MSELoss()

    history = []

    print("Starting RANSCNN-PINN Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_data_loss = 0
        epoch_phys_loss = 0

        for batch in train_loader:
            coords = batch["x"].to(device)
            y_true = batch["y"].to(device)

            optimizer.zero_grad()
            y_pred = model(coords)

            # Data Loss
            loss_data = data_criterion(y_pred, y_true)

            # Physics Loss
            loss_phys = compute_pinn_loss(coords, y_pred)

            loss_total = loss_data + (LAMBDA_PHYS * loss_phys)

            loss_total.backward()
            optimizer.step()

            epoch_data_loss += loss_data.item()
            epoch_phys_loss += loss_phys.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Data MSE: {epoch_data_loss / len(train_loader):.6f}")

        history.append({"epoch": epoch, "data_loss": epoch_data_loss / len(train_loader)})

    torch.save(model.state_dict(), FINAL_MODEL_FILENAME)
    print("Training Complete!")


if __name__ == "__main__":
    main()