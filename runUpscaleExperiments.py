"""
BATCH UPSCALING EXPERIMENT RUNNER
---------------------------------
Function:
    Automates the execution of the Zero-Shot Super-Resolution training script (`train_naca_upscaled.py`)
    across multiple scale factors (e.g., 1.0x, 2.0x).

Key Features:
    1. Iterative Scaling: Sequentially trains models on synthetic grids of increasing density to determines
       if "hallucinating" finer details during training improves accuracy on the original coarse test set.
    2. Timing Metrics: Tracks and reports the wall-clock time for each experiment, helping you weigh
       the computational cost of upscaling against the accuracy gains.
    3. Result Aggregation: Automatically compiles the final "Scale vs. Error" table to identifying the
       point of diminishing returns.
"""

import subprocess
import sys
import os
import csv
import time

# --- Configuration ---
# List of scale factors you want to test
# 1.0 = Baseline (Train 221x51, Test 221x51)
# 2.0 = Train on 442x102, Test on 221x51
# 4.0 = Train on 884x204, Test on 221x51
SCALE_FACTORS_TO_RUN = [1.0, 2.0]
# --- End Configuration ---

# The name of the script you want to run
SCRIPT_TO_RUN = "train_naca_upscaled.py"
SUMMARY_FILENAME = "results/naca_plots_neuralop/naca_upscale_summary.csv"

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

# --- MODIFICATION: Record total time for all experiments ---
all_start_time = time.time()

for scale in SCALE_FACTORS_TO_RUN:
    print(f"--- Starting experiment for scale_factor = {scale} ---")
    print(f"---")

    # --- MODIFICATION: Start timer for this specific run ---
    experiment_start_time = time.time()

    # Create the command to run
    command = [
        python_executable,
        script_path,
        f"--scale_factor={scale}"
    ]

    # Run the script and wait for it to complete
    try:
        subprocess.run(command, check=True)

        # --- MODIFICATION: Stop timer and print elapsed time ---
        experiment_end_time = time.time()
        elapsed_time = experiment_end_time - experiment_start_time
        print(f"\n--- Finished experiment for scale_factor = {scale} (Took {elapsed_time:.2f} seconds) ---")
        # --- END MODIFICATION ---

    except subprocess.CalledProcessError as e:
        print(f"\n--- ERROR during experiment for scale_factor = {scale} ---")
        print(f"Command failed with exit code {e.returncode}")
        print("Stopping batch run.")
        break
    except FileNotFoundError:
        print(f"Error: Could not find script at {script_path}")
        print("Please make sure SCRIPT_TO_RUN is set correctly.")
        break

    time.sleep(1)

# --- MODIFICATION: Print total batch run time ---
all_end_time = time.time()
total_elapsed = all_end_time - all_start_time
print(f"\n---")
print(f"--- All experiments complete. (Total time: {total_elapsed:.2f} seconds) ---")
# --- END MODIFICATION ---


# Read and display summary file
print(f"\n---")
print("--- Final Upscale Experiment Summary ---")
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