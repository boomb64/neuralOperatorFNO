"""
BODY-FITTED WING RECOVERY
-------------------------
Uses contourf for better handling of non-uniform CFD meshes.
Specifically targets the [0, 1] chord range.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from neuralop.models import FNO

# ================= CONFIGURATION =================
DATA_PATH = os.path.join(".", "data", "naca")
MODEL_PATH = os.path.join(".", "model", "naca_final_model_neuralop_manual.pth")

# TIGHT ZOOM: NACA airfoils usually exist between X=0 and X=1
# We will zoom slightly out to see the stagnation point and wake.
X_ZOOM = [-0.1, 1.2]
Y_ZOOM = [-0.3, 0.3]

MODES, WIDTH = 12, 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Load Everything
    X = np.load(os.path.join(DATA_PATH, "NACA_Cylinder_X.npy"))
    Y = np.load(os.path.join(DATA_PATH, "NACA_Cylinder_Y.npy"))
    Q = np.load(os.path.join(DATA_PATH, "NACA_Cylinder_Q.npy"))

    model = FNO(n_modes=(MODES, MODES), hidden_channels=WIDTH, in_channels=2, out_channels=5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    # 2. Pick the sample and Predict
    idx = -1
    x_phys, y_phys = X[idx], Y[idx]

    # Prep input for model
    inp = torch.tensor(np.stack([x_phys, y_phys], axis=0), dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).cpu().numpy()[0]

    # 3. Plotting with Contourf (Smoother for body-fitted meshes)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Ground Truth U-Velocity
    ax0 = axes[0]
    cf0 = ax0.contourf(x_phys, y_phys, Q[idx, 0], levels=50, cmap='jet')
    ax0.set_title("Ground Truth - Wing Detail")
    plt.colorbar(cf0, ax=ax0)

    # Prediction U-Velocity
    ax1 = axes[1]
    cf1 = ax1.contourf(x_phys, y_phys, pred[0], levels=50, cmap='jet')
    ax1.set_title("PINN Prediction - Wing Detail")
    plt.colorbar(cf1, ax=ax1)

    # Apply tight zoom to reveal the wing
    for ax in axes:
        ax.set_xlim(X_ZOOM)
        ax.set_ylim(Y_ZOOM)
        ax.set_aspect('equal')
        # Darken the background to see the mesh boundaries
        ax.set_facecolor('#eeeeee')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()