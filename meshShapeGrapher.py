"""
MESH GEOMETRY VISUALIZER (Sanity Check)
---------------------------------------
Function:
    Reads a single raw Tecplot .dat file and generates a 2D scatter plot of the mesh nodes
    in the X-Z plane.

Key Features:
    1. Coordinate Verification: Specifically plots the X-Z plane to visually confirm the
       airfoil orientation (validating why the previous X-Y interpolation failed).
    2. Geometry Inspection: Allows quick visual verification that the point cloud actually
       resembles an airfoil and isn't corrupted.
    3. Debugging: A standalone utility to check individual files without running the full
       preprocessing pipeline.
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os


# --- This function is copied from preprocess_tecplot.py ---
def parse_tecplot_block(filepath, target_var_names):
    """
    Parses a Tecplot .dat file with BLOCK datapacking.
    Extracts X, Y, Z, and the specified list of target variables.
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
            z_idx = variables.index("Z")  # <-- ADDED
            target_indices = [variables.index(name) for name in target_var_names]
        except ValueError as e:
            print(f"Warning: Could not find all variables in {filepath}. Error: {e}. Skipping.")
            print(f"Available variables are: {variables}")
            print(f"Looking for: {['X', 'Y', 'Z'] + target_var_names}")
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
        z_data = data_matrix[z_idx, :]  # <-- ADDED

        points = np.stack([x_data, y_data, z_data], axis=1)  # Shape (N, 3) <-- CHANGED

        # --- FIX ---
        # Stack target values ONLY if target_indices is not empty
        if not target_indices:
            values = np.array([])  # Return an empty array
        else:
            target_data_list = [data_matrix[idx, :] for idx in target_indices]
            values = np.stack(target_data_list, axis=1)  # Shape (N, C_out)
        # --- END FIX ---

        return points, values

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return None, None


# --- Main plot script ---
def plot_mesh(dat_file_path, output_image_name="airfoil_cross_section_XZ.png"):  # <-- Changed default name
    print(f"Loading mesh points from {dat_file_path}...")

    # We only need X, Y, Z, so we can pass an empty list for target_var_names
    points, _ = parse_tecplot_block(dat_file_path, target_var_names=[])

    if points is None:
        print("Failed to parse file.")
        return

    print(f"Successfully loaded {points.shape[0]} points.")

    x = points[:, 0]  # X coordinate
    z = points[:, 2]  # <-- CHANGED from points[:, 1] (Y) to points[:, 2] (Z)

    print("Generating plot...")
    plt.figure(figsize=(10, 8))
    # Use a scatter plot with very small markers to see the shape
    plt.scatter(x, z, s=0.1)  # <-- CHANGED from (x, y) to (x, z)

    plt.title("Airfoil Mesh Point Cloud (X-Z Cross-Section)")  # <-- CHANGED
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")  # <-- CHANGED
    # 'equal' axis is crucial for seeing the true shape of an airfoil
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(output_image_name)
    print(f"Successfully saved plot to {output_image_name}")
    plt.close()


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Put the full path to ONE of your .dat files here
    FILE_TO_CHECK = r"C:\Users\danie\PycharmProjects\neuralOperatorFNO\data\tecplot_data\Tecplot_AOA_12.000_Beta_.000_Vel_2.820.dat"
    # --- END CONFIGURATION ---

    if not os.path.exists(FILE_TO_CHECK):
        print(f"ERROR: File not found: {FILE_TO_CHECK}")
        print("Please update the FILE_TO_CHECK variable in this script.")
    else:
        plot_mesh(FILE_TO_CHECK)

