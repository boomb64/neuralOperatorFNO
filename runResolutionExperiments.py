"""
OUTDATED: use correctFNOresolutionRUNNER.py

AUTOMATED EXPERIMENT ORCHESTRATOR
---------------------------------
Function:
    A wrapper script that automates the execution of `resolutionTester.py` across a
    defined list of subsampling rates (r=1, 2, 4, 8).

Key Features:
    1. Batch Execution: Runs experiments sequentially, eliminating the need for manual intervention
       between training runs.
    2. Environment Safety: Uses `sys.executable` to ensure the subprocess uses the active
       Python environment (preserving PyTorch dependencies).
    3. Immediate Reporting: Automatically parses and prints the final summary CSV to the console
       so you can see the "Resolution vs. Error" trade-off immediately after the batch finishes.
"""

import subprocess
import sys
import os
import csv
import time

# --- Configuration ---
# List of 'r' values you want to test
# This is the list you requested!
R_VALUES_TO_RUN = [1, 2, 4, 8]
# --- End Configuration ---

# The name of the script you want to run
SCRIPT_TO_RUN = "resolutionTester.py"
SUMMARY_FILENAME = "results/naca_plots_neuralop/naca_experiment_summary.csv"

# Get the path to the current Python executable
python_executable = sys.executable

# Get the directory of this wrapper script to find the target script
base_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(base_dir, SCRIPT_TO_RUN)

# Delete old summary file
summary_filepath = os.path.join(base_dir, SUMMARY_FILENAME)
if os.path.exists(summary_filepath):
    print(f"Removing old summary file: {summary_filepath}")
    os.remove(summary_filepath)

print(f"Using Python interpreter: {python_executable}")
print(f"Target script: {script_path}\n")

for r_val in R_VALUES_TO_RUN:
    print(f"---" * 20)
    print(f"--- Starting experiment for r = {r_val} ---")
    print(f"---" * 20)

    # Create the command to run
    command = [
        python_executable,
        script_path,
        f"--subsample_r={r_val}"
    ]

    # Run the script and wait for it to complete
    try:
        subprocess.run(command, check=True)
        print(f"\n--- Finished experiment for r = {r_val} ---")

    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR during experiment for r = {r_val} ---")
        print(f"Command failed with exit code {e.returncode}")
        print("Stopping batch run.")
        break
    except FileNotFoundError:
        print(f"Error: Could not find script at {script_path}")
        print("Please make sure SCRIPT_TO_RUN is set correctly.")
        break

    time.sleep(1)  # Small pause between runs

print(f"\n---" * 20)
print("--- All experiments complete. ---")

# Read and display summary file
print(f"\n---" * 20)
print("--- Final Experiment Summary ---")
try:
    with open(summary_filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Print header
        print(
            f"{header[0]:<7} | {header[1]:<12} | {header[2]:<10} | {header[3]:<20} | {header[4]:<20} | {header[5]:<21} | {header[6]:<16}")
        print("-" * 115)
        # Print rows
        for row in reader:
            print(
                f"{row[0]:<7} | {row[1]:<12} | {row[2]:<10} | {row[3]:<20} | {row[4]:<20} | {row[5]:<21} | {row[6]:<16}")
except FileNotFoundError:
    print(f"Error: Summary file not found at {summary_filepath}")
except Exception as e:
    print(f"Error reading summary file: {e}")