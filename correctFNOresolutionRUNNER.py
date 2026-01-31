"""
BATCH RESOLUTION EXPERIMENT RUNNER
----------------------------------
Function:
    Automates the "Set and Forget" execution of the Resolution Tester script across multiple
    subsampling rates (e.g., r=2, 4, 8) sequentially.

Key Features:
    1. Workflow Automation: Eliminates the need to manually wait for one training run to finish
       before starting the next.
    2. Environment Safety: Uses `sys.executable` to guarantee that the subprocess runs inside
       your active Virtual Environment (where PyTorch/NeuralOperator are installed).
    3. Error Handling: Stops the chain immediately if a specific resolution run crashes, preventing
       wasted compute time.
"""

import os
import subprocess
import time
import sys

# The name of the script you just saved
# Make sure this matches your actual filename exactly!
SCRIPT_NAME = "resolutionFNOCorrect.py"

# The resolutions you want to test
resolutions = [2, 4, 8]

print(f"--- Starting Batch Experiments for {resolutions} ---")

for r in resolutions:
    print(f"\n\n==================================================")
    print(f"STARTING TRAINING FOR SUBSAMPLE RATE: r={r}")
    print(f"==================================================")

    start_time = time.time()

    # --- FIX IS HERE ---
    # Instead of "python", we use sys.executable.
    # This forces the subprocess to use the VENV python (where torch is installed).
    try:
        subprocess.run([sys.executable, SCRIPT_NAME, "--subsample_r", str(r)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error occurred running r={r}. Stopping experiments.")
        break

    elapsed = time.time() - start_time
    print(f"Finished r={r} in {elapsed / 60:.1f} minutes.")

print("\n\nAll experiments completed.")