import numpy as np
import matplotlib.pyplot as plt

# Load the data
print("Loading data...")
X = np.load("data/naca/NACA_Cylinder_X.npy")
Y = np.load("data/naca/NACA_Cylinder_Y.npy")

# Take the first sample
x_sample = X[0]  # Shape: [221, 51]
y_sample = Y[0]  # Shape: [221, 51]

# Plot 1: The Physical Grid (Zoomed in)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Plot every 5th grid line to see the pattern clearly
plt.plot(x_sample[::5, :].T, y_sample[::5, :].T, 'k-', alpha=0.3, linewidth=0.5) # Streamlines
plt.plot(x_sample[:, ::5], y_sample[:, ::5], 'k-', alpha=0.3, linewidth=0.5)     # Normals
plt.scatter(x_sample, y_sample, s=0.5, c='red') # Plot actual points
plt.title("Physical Grid Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.xlim(-0.5, 1.5) # Zoom in on the airfoil
plt.ylim(-0.5, 0.5)

# Plot 2: Clustering Analysis (Delta X)
# Calculate the distance between adjacent points along the "normal" direction (away from wing)
# In C-grids, index 0 is usually the surface.
dist_away_from_wing = np.sqrt(np.diff(x_sample, axis=1)**2 + np.diff(y_sample, axis=1)**2)
avg_dist = np.mean(dist_away_from_wing, axis=0) # Average along the airfoil length

plt.subplot(1, 2, 2)
plt.plot(avg_dist, '.-')
plt.title("Grid Spacing vs. Distance from Wing")
plt.xlabel("Grid Index j (0=Surface, 50=Farfield)")
plt.ylabel("Physical Distance (meters)")
plt.grid(True)

plt.tight_layout()
plt.show()