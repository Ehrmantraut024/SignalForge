import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 500  # Size of the grid for the heatmap
center = (grid_size // 2, grid_size // 2)  # Center of the antenna
max_distance = 100  # Max distance from the center for visualization

# Create a grid
x = np.linspace(-max_distance, max_distance, grid_size)
y = np.linspace(-max_distance, max_distance, grid_size)
X, Y = np.meshgrid(x, y)

# Calculate distance from the center
R = np.sqrt(X**2 + Y**2)

# Avoid division by zero at the center
R[R == 0] = np.nan

# Signal strength following the inverse square law
signal_strength = 1 / R**2

# Normalize the signal strength for better visualization
normalized_strength = np.log10(signal_strength + 1e-10)  # Adding a small value to avoid log(0)

# Plot the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(
    normalized_strength, 
    extent=(-max_distance, max_distance, -max_distance, max_distance), 
    origin='lower', 
    cmap='hot'
)
plt.colorbar(label='Log(Signal Strength)')
plt.title('Isotropic Antenna Signal Strength (Inverse Square Law)')
plt.xlabel('X Distance')
plt.ylabel('Y Distance')
plt.grid(False)
plt.show()
