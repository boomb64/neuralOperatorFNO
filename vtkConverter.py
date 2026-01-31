"""
OUTDATED: This was for a weird data format that trisha sent me once, but now it no longer is needed.

VTK DATA CONVERTER (OpenFOAM/ParaView -> FNO Tensor)
----------------------------------------------------
Function:
    Ingests simulation data in the standard VTK format (common outputs from OpenFOAM or
    ParaView exports) and interpolates it onto a standardized uniform grid for FNO training.

Key Features:
    1. Native Format Support: Reads .vtk files directly using PyVista, preserving the original
       mesh topology before interpolation.
    2. Domain Locking: Uses `FIXED_BOUNDS` to force every simulation—regardless of its
       original mesh density—into the exact same physical coordinate box (e.g., -10 to 30).
    3. Tensor Generation: Outputs the specific (3, 256, 64) numpy arrays (Velocity X, Velocity Y, Pressure)
       required by the FNO data loader.
"""

import pyvista as pv
import numpy as np
import os

# --- 1. Configuration ---

# === CRITICAL: UPDATE THESE VALUES ===
# Use the *exact same* bounds as in the CSV script.
# Format: [X_min, X_max, Y_min, Y_max]
# Example values (replace with your real ones):
FIXED_BOUNDS = [-10.0, 30.0, -20.0, 20.0]
# =======================================

VTK_FILE_PATH = 'Naca2D_Re3.6.vtk' # Path to your VTK file
OUTPUT_DIR = 'processed_fno_data'
OUTPUT_PREFIX = 'naca_vtk_sample_01' # Base name for the output files

# Define the new uniform grid you want to interpolate onto
GRID_RESOLUTION_X = 256
GRID_RESOLUTION_Y = 64

# List of variable names to extract from the VTK file.
# These names MUST match the data arrays in the VTK file.
# Run this script once to see the "Available data arrays" print,
# then update this list accordingly.
VARIABLES_TO_EXTRACT = ['Vx', 'Vy', 'Cp'] # Example: X-Vel, Y-Vel, Pressure Coeff.
# The FNO script expects (U, V, P), so make sure the order is consistent.

# --- 2. Load the VTK File ---

print(f"Loading VTK file: {VTK_FILE_PATH}...")
try:
    mesh = pv.read(VTK_FILE_PATH)
except FileNotFoundError:
    print(f"Error: File not found at {VTK_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error reading VTK file: {e}")
    exit()

print("VTK file loaded successfully.")
print("Available data arrays:", list(mesh.point_data.keys()))
# --- Check available arrays printed here and update VARIABLES_TO_EXTRACT ---

# --- 3. Create the New Uniform Grid ---

print(f"Using FIXED bounds: {FIXED_BOUNDS}")

# --- MODIFICATION START ---
# The function `pv.create_uniform_grid` and `pv.UniformGrid` constructor
# are not found in your pyvista version.
# We will create a pv.ImageData object (the base class) directly.

# Get dimensions and bounds
nx, ny = GRID_RESOLUTION_X, GRID_RESOLUTION_Y
x_min, x_max, y_min, y_max = FIXED_BOUNDS

# Calculate origin (the bottom-left corner)
origin = (x_min, y_min, 0.0)

# Calculate spacing between points
# Handle potential division by zero if a dimension has only 1 point
spacing_x = (x_max - x_min) / (nx - 1) if nx > 1 else 0.0
spacing_y = (y_max - y_min) / (ny - 1) if ny > 1 else 0.0
spacing = (spacing_x, spacing_y, 0.0) # Z-spacing is 0 for a 2D plane

# Create the grid using the ImageData class
new_grid = pv.ImageData()
new_grid.dimensions = (nx, ny, 1)
new_grid.spacing = spacing
new_grid.origin = origin
# --- MODIFICATION END ---

print(f"Created new uniform grid with dimensions {GRID_RESOLUTION_X}x{GRID_RESOLUTION_Y}.")

# --- 4. Interpolate (Sample) Data onto the New Grid ---

print("Interpolating data from VTK file onto the new uniform grid...")
interpolated_grid = new_grid.interpolate(mesh, radius=mesh.length * 0.05)
print("Interpolation complete.")

# --- 5. Extract and Save Data as .npy ---

os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    # 1. Save Coordinates (from the new uniform grid)
    points_xyz = interpolated_grid.points.reshape((GRID_RESOLUTION_X, GRID_RESOLUTION_Y, 1, 3))
    points_xyz = points_xyz.squeeze(axis=2) # Shape: (256, 64, 3)

    x_coords = points_xyz[:, :, 0] # Shape: (256, 64)
    y_coords = points_xyz[:, :, 1] # Shape: (256, 64)

    # Save the grid coordinates. You only need to do this ONCE.
    # All subsequent samples will use this exact grid.
    if not os.path.exists(os.path.join(OUTPUT_DIR, "uniform_grid_X.npy")):
        np.save(os.path.join(OUTPUT_DIR, "uniform_grid_X.npy"), x_coords)
        np.save(os.path.join(OUTPUT_DIR, "uniform_grid_Y.npy"), y_coords)
        print(f"Saved uniform_grid_X.npy and uniform_grid_Y.npy")


    # 2. Save Solution Variables
    solution_data_list = []
    for var_name in VARIABLES_TO_EXTRACT:
        if var_name not in interpolated_grid.point_data:
            print(f"Warning: Variable '{var_name}' not found in interpolated data. Skipping.")
            continue

        data_flat = interpolated_grid.point_data[var_name]
        data_grid = data_flat.reshape((GRID_RESOLUTION_X, GRID_RESOLUTION_Y))
        solution_data_list.append(data_grid)
        print(f"Extracted '{var_name}'")

    if not solution_data_list:
        print("Error: No solution variables were extracted. Check VARIABLES_TO_EXTRACT.")
    else:
        solution_stack = np.stack(solution_data_list, axis=0)
        np.save(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_Q.npy"), solution_stack)
        print(f"Saved solution data (Q.npy) with shape {solution_stack.shape}")

    print(f"\nProcessing complete. Files saved in '{OUTPUT_DIR}' directory.")

except Exception as e:
    print(f"\nAn error occurred during extraction or saving: {e}")