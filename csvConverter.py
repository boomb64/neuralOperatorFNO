"""
UNSTRUCTURED DATA CONVERTER (Excel -> FNO Tensor)
-------------------------------------------------
Function:
    Ingests raw CFD results exported as Excel/CSV (unstructured point clouds) and
    interpolates them onto a standardized uniform grid for FNO training.

Key Features:
    1. Mesh-to-Grid Resampling: Uses PyVista to convert scattered simulation nodes into
       a structured 256x64 image-like format.
    2. Domain Standardization: Enforces fixed physical bounds (FIXED_BOUNDS) to ensure
       that geometries from different simulations align perfectly in the input tensor.
    3. Format Bridging: Bridges the gap between commercial CFD tools (Ansys/Star-CCM+)
       and the .npy tensor format required by PyTorch.
"""

import pyvista as pv
import numpy as np
import pandas as pd  # Use pandas to read Excel
import os

# --- 1. Configuration ---
FIXED_BOUNDS = [-10.0, 30.0, -20.0, 20.0]

# --- MODIFICATION ---
# Path to your XLSX file
XLSX_FILE_PATH = 'NACA_0015_2D.xlsx'
# --- END MODIFICATION ---

OUTPUT_DIR = 'processed_fno_data'
OUTPUT_PREFIX = 'naca_xlsx_sample_01'  # Base name for the output files

# Define the new uniform grid you want to interpolate onto
GRID_RESOLUTION_X = 256
GRID_RESOLUTION_Y = 64

# List of variable names to extract from the Excel file.
# These names MUST match the headers in the Excel sheet.
VARIABLES_TO_EXTRACT = ['Vx', 'Vy', 'Cp']  # Example: X-Vel, Y-Vel, Pressure Coeff.
# The FNO script expects (U, V, P), so make sure the order is consistent.

# Column names for the coordinates in the Excel file.
# Check your file for the correct headers.
COORD_COLS = ['X', 'Y', 'Points:2']

# --- 2. Load the XLSX File ---

print(f"Loading XLSX file: {XLSX_FILE_PATH}...")
try:
    # --- MODIFICATION ---
    # Use pd.read_excel() to read the .xlsx file
    # It defaults to reading the first sheet (Sheet1)
    df = pd.read_excel(XLSX_FILE_PATH)
    # --- END MODIFICATION ---
except FileNotFoundError:
    print(f"Error: File not found at {XLSX_FILE_PATH}")
    print("Please make sure the script is in the same directory as the XLSX, or update XLSX_FILE_PATH.")
    exit()
except ImportError:
    print("Error: `openpyxl` library not found. Please install it: pip install openpyxl")
    exit()
except Exception as e:
    print(f"Error reading XLSX file: {e}")
    exit()

print("XLSX file loaded successfully.")
print("Available data columns (headers):", list(df.columns))

# --- 3. Create PyVista Mesh from Data ---

try:
    # Get the point coordinates from the dataframe
    points = df[COORD_COLS].values

    # Create an unstructured mesh (PolyData) from these points
    mesh = pv.PolyData(points)

    # Add the solution variables from the dataframe to the mesh as point data
    for var_name in VARIABLES_TO_EXTRACT:
        if var_name not in df.columns:
            print(f"Warning: Variable '{var_name}' not found in XLSX columns. Skipping.")
            continue
        mesh[var_name] = df[var_name].values
        print(f"Added data array '{var_name}' to mesh.")

except KeyError:
    print(f"Error: Could not find coordinate columns {COORD_COLS} in the XLSX.")
    print("Please check the COORD_COLS variable in this script.")
    exit()

# --- 4. Create the New Uniform Grid ---

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
spacing = (spacing_x, spacing_y, 0.0)  # Z-spacing is 0 for a 2D plane

# Create the grid using the ImageData class
new_grid = pv.ImageData()
new_grid.dimensions = (nx, ny, 1)
new_grid.spacing = spacing
new_grid.origin = origin
# --- MODIFICATION END ---

print(f"Created new uniform grid with dimensions {GRID_RESOLUTION_X}x{GRID_RESOLUTION_Y}.")

# --- 5. Interpolate (Sample) Data onto the New Grid ---

print("Interpolating data from mesh onto the new uniform grid...")
# This function samples the data from the original `mesh` at the
# locations of the `new_grid` points.
# We set a small radius to find nearby points for interpolation.
interpolated_grid = new_grid.interpolate(mesh, radius=mesh.length * 0.05)
print("Interpolation complete.")

# --- 6. Extract and Save Data as .npy ---

os.makedirs(OUTPUT_DIR, exist_ok=True)
try:
    # 1. Save Coordinates (from the new uniform grid)
    points_xyz = interpolated_grid.points.reshape((GRID_RESOLUTION_X, GRID_RESOLUTION_Y, 1, 3))
    points_xyz = points_xyz.squeeze(axis=2)  # Shape: (256, 64, 3)

    x_coords = points_xyz[:, :, 0]  # Shape: (256, 64)
    y_coords = points_xyz[:, :, 1]  # Shape: (256, 64)

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

        # Reshape to (NX, NY)
        data_grid = data_flat.reshape((GRID_RESOLUTION_X, GRID_RESOLUTION_Y))
        solution_data_list.append(data_grid)
        print(f"Extracted '{var_name}'")

    if not solution_data_list:
        print("Error: No solution variables were extracted. Check VARIABLES_TO_EXTRACT.")
    else:
        # Stack the variables along a new "channels" axis
        # Shape: (num_variables, GRID_RESOLUTION_X, GRID_RESOLUTION_Y)
        solution_stack = np.stack(solution_data_list, axis=0)

        np.save(os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_Q.npy"), solution_stack)
        print(f"Saved solution data (Q.npy) with shape {solution_stack.shape}")

    print(f"\nProcessing complete. Files saved in '{OUTPUT_DIR}' directory.")

except Exception as e:
    print(f"\nAn error occurred during extraction or saving: {e}")