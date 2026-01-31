"""
Not super used, but not outdated. I wrote it to give Trisha scenarios to generate,
but she just used flighstream functions and a range to generate scenarios.

EXPERIMENT DESIGN GENERATOR (LHS)
---------------------------------
Function:
    Generates a statistically balanced schedule of simulation parameters (Angle of Attack & Reynolds Number)
    for the CFD solver to execute.

Key Features:
    1. Efficient Space Filling: Uses Latin Hypercube Sampling (LHS) instead of random sampling to ensure
       the entire parameter space is covered evenly without clustering.
    2. Reproducibility: Uses a fixed random seed (42) so the experiment design is deterministic and repeatable.
    3. Pipeline Input: Produces the master 'cfd_scenarios.csv' file that drives the batch CFD simulations.
"""

import numpy as np
from scipy.stats import qmc # qmc is for Quasi-Monte Carlo methods, including LHS

# --- 1. Define Parameters ---

# Total number of simulations to run (1000 for training, 200 for testing)
N_SAMPLES = 1200

# Number of parameters we are varying (AoA and Re)
N_DIMENSIONS = 2

# Parameter ranges
MIN_AOA = 0.0
MAX_AOA = 12.0
MIN_RE = 2_000_000.0
MAX_RE = 6_000_000.0

# Output filename
OUTPUT_FILE = 'cfd_scenarios.csv'

# Set a random seed for reproducibility.
# Using the same seed will always generate the exact same list of scenarios.
RANDOM_SEED = 42

# --- 2. Generate LHS Samples ---

print(f"Generating {N_SAMPLES} scenarios using Latin Hypercube Sampling...")
print(f"  Angle of Attack range: {MIN_AOA} to {MAX_AOA} degrees")
print(f"  Reynolds Number range: {MIN_RE:,.0f} to {MAX_RE:,.0f}")

# 1. Initialize the LHS sampler
#    'd' is the number of dimensions (parameters)
#    'seed' ensures you can get the same "random" list again
sampler = qmc.LatinHypercube(d=N_DIMENSIONS, seed=RANDOM_SEED)

# 2. Generate the samples in a "unit cube" (all values are between 0 and 1)
unit_samples = sampler.random(n=N_SAMPLES)

# 3. Define the lower and upper bounds for scaling
l_bounds = [MIN_AOA, MIN_RE]
u_bounds = [MAX_AOA, MAX_RE]

# 4. Scale the unit cube samples to our real parameter ranges
scaled_samples = qmc.scale(unit_samples, l_bounds, u_bounds)

# --- 3. Save to File ---

# We'll save as a CSV (Comma Separated Values) file for easy use
header = "Scenario_ID,Angle_of_Attack_deg,Reynolds_Number"
comments = '' # We don't want a '#' prefix on the header

# Use numpy.savetxt for easy and clean formatting
# We'll create an array with the scenario ID (1, 2, 3...)
scenario_ids = np.arange(1, N_SAMPLES + 1).reshape(-1, 1)

# Combine scenario IDs with the scaled samples
# final_data shape will be (1200, 3)
final_data = np.hstack((scenario_ids, scaled_samples))

# Save the data
# fmt specifies the format:
#   %d    = integer (for Scenario_ID)
#   %.4f  = float with 4 decimal places (for AoA)
#   %.0f  = float with 0 decimal places (for Reynolds Number)
np.savetxt(
    OUTPUT_FILE,
    final_data,
    delimiter=',',
    header=header,
    comments=comments,
    fmt=['%d', '%.4f', '%.0f']
)

print(f"\nSuccessfully generated and saved {N_SAMPLES} scenarios to {OUTPUT_FILE}")

# --- 4. Show a preview ---
print("\n--- Preview of first 5 scenarios: ---")
print(header)
for i in range(5):
    print(f"{int(final_data[i, 0])},{final_data[i, 1]:.4f},{final_data[i, 2]:.0f}")
