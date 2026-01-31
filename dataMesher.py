"""
OLD VERSION: ONLY USED ON OLD DATA. USE trishaDataPreprcessor.py

STRUCTURED GRID INTERPOLATOR (Tecplot -> .npy)
----------------------------------------------
Function:
    Parses unstructured Tecplot BLOCK data (X, Y, Z, Cp) and interpolates it onto
    a user-defined Uniform Grid (X-Z Plane) for CNN/FNO training.

Key Features:
    1. Dimension Handling: Specifically configured to extract the X-Z plane (common in 2D CFD),
       ignoring the zeroed Y-dimension to prevent "flat simplex" interpolation errors.
    2. Interpolation: Uses `scipy.interpolate.griddata` (Linear) to map scattered CFD nodes
       onto a clean 200x200 pixel grid.
    3. Output: Produces stacked .npy tensors [N, H, W] for the Geometry (X, Z) and the
       Solution Field (Q/Cp).
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from scipy.interpolate import griddata
from glob import glob

# --- USER CONFIGURATION ---
# 1. Path to your raw .dat files
SOURCE_DATA_DIR = r"C:\Users\danie\PycharmProjects\neuralOperatorFNO\data\tecplot_data"

# 2. Path to save the processed .npy files
SAVE_DIR = r"./data/tecplot_processed_simple_XZ" # <-- Note: Changed folder name

# 3. Output variable name (must be exact match from .dat file VARIABLES list)
OUTPUT_VARIABLE_NAME = "Cp"

# 4. Target grid resolution
GRID_RESOLUTION_X = 200
GRID_RESOLUTION_Y = 200 # <-- This is now for Z, but we'll keep the var name simple

# 5. Define the interpolation domain (X-Z plane)
#    You MUST adjust these to fit your data!
#    Run check_mesh.py to see your data's X and Z ranges.
MIN_X = -1.0
MAX_X = 2.0
MIN_Z = -1.5 # <-- CHANGED from MIN_Y
MAX_Z = 1.5 # <-- CHANGED from MAX_Y

# --- The coordinate names to extract and interpolate onto ---
TARGET_COORD_NAMES = ["X", "Z"] # <-- CHANGED from ["X", "Y"]

# --- Names for the final saved files ---
SAVE_X_FILENAME = "NACA_Cylinder_X.npy"
SAVE_Z_FILENAME = "NACA_Cylinder_Z.npy" # <-- CHANGED from SAVE_Y_FILENAME
SAVE_Q_FILENAME = "NACA_Cylinder_Q.npy"
# --- END USER CONFIGURATION ---

def parse_tecplot_block(filepath, target_var_names):
    """
    Parses a Tecplot .dat file with BLOCK datapacking.
    Extracts X, Y, Z and the specified list of target variables.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        variables = []
        num_nodes = 0
        data_start_line = -1

        for i, line in enumerate(lines):
            line = line.strip()
            if line.upper().startswith("VARIABLES"):
                var_str = line.split("=")[1].strip()
                variables = [v.strip().strip('"') for v in var_str.split(',')]

            elif line.upper().startswith("ZONE"):
                nodes_match = re.search(r'NODES=(\d+)', line.upper())
                if nodes_match:
                    num_nodes = int(nodes_match.group(1))
                data_start_line = i + 1
                break

        if not variables or num_nodes == 0 or data_start_line == -1:
            print(f"Warning: Could not parse header for {filepath}. Skipping.")
            return None, None

        try:
            x_idx = variables.index("X")
            y_idx = variables.index("Y")
            z_idx = variables.index("Z") # <-- ADDED
            target_indices = [variables.index(name) for name in target_var_names]
        except ValueError as e:
            print(f"Warning: Could not find all variables in {filepath}. Error: {e}. Skipping.")
            print(f"Available variables are: {variables}")
            return None, None

        all_data = []
        data_lines = lines[data_start_line:]
        for line in data_lines:
            all_data.extend([float(v) for v in line.strip().split()])

        all_data = np.array(all_data)
        expected_size = num_nodes * len(variables)
        if all_data.size < expected_size:
            print(f"Warning: Data block in {filepath} is smaller than expected. Skipping.")
            return None, None

        all_data = all_data[:expected_size]

        try:
            data_matrix = all_data.reshape((len(variables), num_nodes))
        except ValueError:
            print(f"Warning: Data block in {filepath} has incorrect shape. Skipping.")
            return None, None

        x_data = data_matrix[x_idx, :]
        y_data = data_matrix[y_idx, :]
        z_data = data_matrix[z_idx, :] # <-- ADDED

        # Shape (N, 3)
        points = np.stack([x_data, y_data, z_data], axis=1) # <-- CHANGED

        if not target_indices:
            values = np.array([])
        else:
            target_data_list = [data_matrix[idx, :] for idx in target_indices]
            values = np.stack(target_data_list, axis=1)

        return points, values

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None, None

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    dat_files = glob(os.path.join(SOURCE_DATA_DIR, "*.dat"))
    if not dat_files:
        print(f"Error: No .dat files found in {SOURCE_DATA_DIR}")
        return

    print(f"Found {len(dat_files)} .dat files.")

    # 1. Create the target grid
    grid_x_vec = np.linspace(MIN_X, MAX_X, GRID_RESOLUTION_X)
    grid_z_vec = np.linspace(MIN_Z, MAX_Z, GRID_RESOLUTION_Y) # <-- Use Z
    grid_x, grid_z = np.meshgrid(grid_x_vec, grid_z_vec) # <-- Use Z

    all_q_data = []

    for filepath in dat_files:
        print(f"Processing {os.path.basename(filepath)}...")

        # Parse X,Y,Z points and the target output variable (e.g., "Cp")
        points_xyz, values = parse_tecplot_block(filepath, target_var_names=[OUTPUT_VARIABLE_NAME])

        if points_xyz is None or values.ndim < 2 or values.shape[1] == 0:
            print(f"Skipping file {filepath} due to parsing error or no data.")
            continue

        # Get only the X-Z coordinates for interpolation
        points_xz = points_xyz[:, [0, 2]] # <-- Shape (N, 2) with X and Z
        values_cp = values[:, 0] # Shape (N,)

        # 2. Interpolate Q (output) onto the new grid
        interpolated_grid = griddata(
            points_xz,       # Unstructured (X, Z)
            values_cp,       # Unstructured Value (e.g., 'Cp')
            (grid_x, grid_z),# New structured (X, Z)
            method='linear',
            fill_value=0.0
        )

        all_q_data.append(interpolated_grid)

    if not all_q_data:
        print("Error: No data was successfully processed.")
        return

    # 3. Stack and save
    # The training script expects (N, H, W)

    # Create N copies of the grid_x and grid_z for stacking
    # grid_x and grid_z have shape (H, W)
    all_x_data = np.stack([grid_x] * len(all_q_data), axis=0)
    all_z_data = np.stack([grid_z] * len(all_q_data), axis=0) # <-- Use Z
    all_q_data = np.stack(all_q_data, axis=0)

    print(f"Final X shape: {all_x_data.shape}")
    print(f"Final Z shape: {all_z_data.shape}")
    print(f"Final Q shape: {all_q_data.shape}")

    np.save(os.path.join(SAVE_DIR, SAVE_X_FILENAME), all_x_data)
    np.save(os.path.join(SAVE_DIR, SAVE_Z_FILENAME), all_z_data) # <-- Use Z
    np.save(os.path.join(SAVE_DIR, SAVE_Q_FILENAME), all_q_data)

    print(f"\nSuccessfully saved {len(all_q_data)} samples to {SAVE_DIR}")

if __name__ == "__main__":
    main()

